/**********************************************************************
* This example aims to demonstrate how to use p-FEM using the NeoPZ   *
* library. It solves the equation div(-k grad(p)) = f for the         *
* variable p using a Continuous Galerkin formulation, i.e., using     *
* traditional H1 elements.                                            *
* The domain is a unit square ranging from (0,0) to (1,1), k=1 and    *
* dirichlet homogeneous conditions are imposed in all boundaries.     *
**********************************************************************/

#include <TPZGmshReader.h>
#include <pzgmesh.h>
#include <pzanalysis.h>
#include <TPZMatLaplacian.h>
#include <pzbndcond.h>
#include <pzstepsolver.h>
#ifdef USING_MKL
#include <StrMatrix/TPZSSpStructMatrix.h>
#else
#include <StrMatrix/pzskylstrmatrix.h>
#endif
#include <Pre/pzgengrid.h>
#include <pzintel.h>

#include <string>


enum EElementType {
    ETriangular = 0, ESquare = 1, ETrapezoidal = 2
};

/**
 * @brief Routine from creating the geometric mesh of the domain. It is a square with nodes
 * (0,0), (1,0), (1,1), (0,1).
 * @param dim dimension of the problem
 * @param nelx number of divisions on the x direction
 * @param nely number of divisions on the y direction
 * @param meshType defines with elements will be created (triangular, square, trapezoidal)
 * @param matIds store the material identifiers
 */
TPZGeoMesh *CreateGeoMesh(const int dim, int nelx, int nely, EElementType meshType, TPZVec<int> &matIds);


/**
* Generates a computational mesh that implements the problem to be solved
*/
static TPZCompMesh *CreateCompMesh(TPZGeoMesh *gmesh, const TPZVec<int> &matIds, const int initialPOrder);

/**
 * This method will perform the P refinement on certain elements.
 * These elements are the ones whose error exceed a given percentage of the maximum error in the mesh.
 */
void PerformPRefinement(TPZCompMesh *cMesh, TPZAnalysis &an, const REAL errorPercentage);

int main(int argc, char **argv)
{
    constexpr int numthreads{8};//number of threads to be used throughout the program
    #ifdef USING_MKL
    mkl_set_dynamic(0); // disable automatic adjustment of the number of threads
    mkl_set_num_threads(numthreads);
    #endif

    constexpr int dim{2};//physical dimension of the problem
    constexpr int nDiv{50};//number of divisions of each direction (x, y) of the domain
    constexpr int initialPOrder{1};//initial polynomial order
    //this will set how many rounds of refinements will be performed
    constexpr int nPRefinements{10};
    //once the element with the maximum error is found, elements with errors bigger than
    //the following percentage will be refined as well
    constexpr REAL errorPercentage{0.3};
    std::string plotfile= "solution.vtk";//where to print the vtk files
    constexpr int postProcessResolution{2};

    /** In NeoPZ, the TPZMaterial classes are used to implement the weak statement of the differential equation,
     * along with setting up the constitutive parameters of each region of the domain. See the method CreateCompMesh
     * in this file for an example.
     * The material ids are identifiers used in NeoPZ to identify different domain's regions/properties.
     */
    TPZVec<int> matIdVec;
    TPZGeoMesh *gMesh = CreateGeoMesh(dim,nDiv,nDiv,ETriangular,matIdVec);
    //creates computational mesh
    TPZCompMesh *cMesh = CreateCompMesh(gMesh,matIdVec,initialPOrder);

    //Setting up the analysis object
    constexpr bool optimizeBandwidth{true};
    TPZAnalysis an(cMesh, optimizeBandwidth); //Creates the object that will manage the analysis of the problem
    {
        //The TPZStructMatrix classes provide an interface between the linear algebra aspects of the library and
        //the Finite Element ones. Their specification (TPZSymetricSpStructMatrix, TPZSkylineStructMatrix, etc) are
        //also used to define the storage format for the matrices. In this example, the first one is the
        //CSR sparse matrix storage, and the second one the skyline, both in their symmetric versions.



        //I highly recommend running this program using the MKL libraries, the solving process will be
        //significantly faster.
#ifdef USING_MKL
        TPZSymetricSpStructMatrix matskl(cMesh);
#else
        TPZSkylineStructMatrix matskl(cMesh);
#endif
        matskl.SetNumThreads(numthreads);
        an.SetStructuralMatrix(matskl);
    }

    //setting solver to be used. ELDLt will default to Pardiso if USING_MKL is enabled.
    TPZStepSolver<STATE> step;
    step.SetDirect(ELDLt);
    an.SetSolver(step);

    //setting reference solution
    auto exactSolution = [] (const TPZVec<REAL> &pt, TPZVec<STATE> &sol, TPZFMatrix<STATE> &solDx){
        const auto x = pt[0], y = pt[1];
        sol[0] = std::pow(10,-std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*(-2*M_PI + 15*x);
        solDx(0,0) = -3*std::pow(2,-8*std::pow(M_PI,2) + 60*M_PI*(x + y) - 225*(std::pow(x,2) + std::pow(y,2)))
                *std::pow(5,1 - 8*std::pow(M_PI,2) + 60*M_PI*(x + y) - 225*(std::pow(x,2) + std::pow(y,2)))
                *(-1 + 2*std::pow(2*M_PI - 15*x,2)*std::log(10));
        solDx(1,0) = -3*std::pow(10,1 - 8*std::pow(M_PI,2) + 60*M_PI*(x + y) - 225*(std::pow(x,2) + std::pow(y,2)))
                *(2*M_PI - 15*x)*(2*M_PI - 15*y)*std::log(10);
    };
    an.SetExact(exactSolution);
    an.SetThreadsForError(numthreads);

    //setting variables for post processing
    TPZStack<std::string> scalnames, vecnames;
    scalnames.Push("Pressure");//print the state variable
    scalnames.Push("POrder");//print the polynomial order of each element
    scalnames.Push("Error");//print the error of each element
    //resize the matrix that will store the error for each element
    cMesh->ElementSolution().Resize(cMesh->NElements(),3);
    for(auto it = 0 ; it < nPRefinements; it++){
        std::cout<<"============================"<<std::endl;
        std::cout<<"\tIteration "<<it+1<<" out of "<<nPRefinements<<std::endl;
        std::cout<<"\tAssemble matrix with NDoF = "<<cMesh->NEquations()<<"."<<std::endl;
        an.SetCompMesh(cMesh,optimizeBandwidth);
        an.Assemble(); //Assembles the global stiffness matrix (and load vector)
        std::cout<<"\tAssemble finished."<<std::endl;
        an.Solve();

        std::cout<<"\tCalculating errors..."<<std::endl;
        TPZVec<REAL> errorVec(3,0);
        an.PostProcessError(errorVec,true);
        std::cout<<"############"<<std::endl;

        std::cout<<"\tPost processing..."<<std::endl;
        an.DefineGraphMesh(dim, scalnames, vecnames, plotfile);
        an.SetStep(it);
        an.PostProcess(postProcessResolution);
        std::cout<<"\tPost processing finished."<<std::endl;
        if(it < nPRefinements - 1){
            PerformPRefinement(cMesh, an, errorPercentage);
        }
    }
    delete cMesh;
    delete gMesh;
    return 0;
}

TPZGeoMesh *CreateGeoMesh(const int dim, int nelx, int nely, EElementType meshType, TPZVec<int> &matIds) {
    //Creating geometric mesh, nodes and elements.
    //Including nodes and elements in the mesh object:
    TPZGeoMesh *gmesh = new TPZGeoMesh();
    gmesh->SetDimension(dim);

    //Auxiliary vector to store coordinates:
    TPZVec<REAL> coord1(3, 0.);
    TPZVec<REAL> coord2(3, 0.);
    coord1[0] = 0;coord1[1] = 0;coord1[2] = 0;
    coord2[0] = 1;coord2[1] = 1;coord2[2] = 0;

    TPZManVector<int> nelem(2, 1);
    nelem[0] = nelx;
    nelem[1] = nely;

    TPZGenGrid gengrid(nelem, coord1, coord2);

    switch (meshType) {
        case ESquare:
            gengrid.SetElementType(EQuadrilateral);
            break;
        case ETriangular:
            gengrid.SetElementType(ETriangle);
            break;
        case ETrapezoidal:
            gengrid.SetDistortion(0.25);
            break;
    }
    constexpr int matIdDomain = 1, matIdBoundary = 2;
    gengrid.Read(gmesh, matIdDomain);
    gengrid.SetBC(gmesh, 4, matIdBoundary);
    gengrid.SetBC(gmesh, 5, matIdBoundary);
    gengrid.SetBC(gmesh, 6, matIdBoundary);
    gengrid.SetBC(gmesh, 7, matIdBoundary);

    gmesh->BuildConnectivity();

    {
        TPZCheckGeom check(gmesh);
        check.CheckUniqueId();
    }
    matIds.Resize(2);
    matIds[0] = matIdDomain;
    matIds[1] = matIdBoundary;
    //Printing geometric mesh:

    //ofstream bf("before.vtk");
    //TPZVTKGeoMesh::PrintGMeshVTK(gmesh, bf);
    return gmesh;
}

TPZCompMesh *CreateCompMesh(TPZGeoMesh *gmesh, const TPZVec<int> &matIds, const int initialPOrder){
    TPZCompMesh *cmesh = new TPZCompMesh(gmesh);

    //Definition of the approximation space
    const int dim = gmesh->Dimension();
    cmesh->SetDefaultOrder(initialPOrder);
    cmesh->SetDimModel(dim);


    const int matId = matIds[0];
    constexpr REAL perm{1};

    //Inserting material
    TPZMatLaplacian * mat = new TPZMatLaplacian(matId, dim);
    mat->SetPermeability(perm);
//    mat->SetNonSymmetric();
//    mat->SetNoPenalty();
    mat->SetSymmetric();


    auto forcingFunction = [](const TPZVec<REAL>& pt, TPZVec<STATE> &result){
        REAL x = pt[0];
        REAL y = pt[1];
        result[0] = 9*std::pow(2,1 - std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*
                std::pow(5,2 - std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*
                (-2*M_PI + 15*x)*std::log(2) + 9*std::pow(2,1 - std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*
                std::pow(5,2 - std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*(-2*M_PI + 15*x)*
                std::log(5) + 9*std::pow(2,1 - std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*
                std::pow(5,2 - std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*
                (-2*M_PI + 15*x)*std::log(10) + 9*std::pow(10,2 - std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*(-2*M_PI + 15*x)*std::log(10) -
                9*std::pow(10,2 - std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*std::pow(-2*M_PI + 15*x,3)*std::pow(std::log(10),2) -
                9*std::pow(10,2 - std::pow(-2*M_PI + 15*x,2) - std::pow(-2*M_PI + 15*y,2))*(-2*M_PI + 15*x)*std::pow(-2*M_PI + 15*y,2)*std::pow(std::log(10),2);
    };
    constexpr int pOrderForcingFunction{10};
    mat->SetForcingFunction(forcingFunction,pOrderForcingFunction);

    //Inserting volumetric materials objects
    cmesh->InsertMaterialObject(mat);

    //Boundary conditions
    constexpr int dirichlet = 0;
    constexpr int neumann = 1;
    TPZFMatrix<STATE> val1(1,1,0.0);
    TPZFMatrix<STATE> val2(1,1,0.0);

    const int &matIdBc1 = matIds[1];
    val2(0,0)=0.0;
    auto bc1 = mat->CreateBC(mat, matIdBc1, dirichlet, val1, val2);
    cmesh->InsertMaterialObject(bc1);

    cmesh->SetAllCreateFunctionsContinuous();//set H1 approximation space
    cmesh->AutoBuild();
    cmesh->AdjustBoundaryElements();
    cmesh->CleanUpUnconnectedNodes();
    return cmesh;
}

void PerformPRefinement(TPZCompMesh *cmesh, TPZAnalysis &an,
                        const REAL errorPercentage) {
    std::cout<<"\tPerforming p-refinement..."<<std::endl;
    const auto nElems = cmesh->Reference()->NElements();
    // Iterates through element errors to get the maximum value
    REAL maxError = -1;
    for (int64_t iel = 0; iel < nElems; iel++) {
        TPZCompEl *cel = cmesh->ElementVec()[iel];
        if (!cel) continue;
        if (cel->Dimension() != cmesh->Dimension()) continue;
        REAL elementError = cmesh->ElementSolution()(iel, 0);
        if (elementError > maxError) {
            maxError = elementError;
        }
    }
    std::cout<<"\tMax error found (in one element): "<<maxError<<std::endl;
    // Refines elements which errors are bigger than 30% of the maximum error
    REAL threshold = errorPercentage * maxError;
    int count = 0;
    for (int64_t iel = 0; iel < nElems; iel++) {
        auto cel =  dynamic_cast<TPZInterpolationSpace *> (cmesh->Element(iel));
        if (!cel || cel->Dimension() != cmesh->Dimension()) continue;
        REAL elementError = cmesh->ElementSolution()(iel, 0);
        if (elementError > threshold) {
            const int currentPorder = cel->GetPreferredOrder();
            cel->PRefine(currentPorder+1);
            count++;
        }
    }
    std::cout<<"\t"<<count<<" elements were refined in this step."<<std::endl;
    cmesh->AdjustBoundaryElements();
    cmesh->CleanUpUnconnectedNodes();
    cmesh->ExpandSolution();
    const auto neqNew = cmesh->NEquations();
    an.StructMatrix()->EquationFilter().SetNumEq(neqNew);
}