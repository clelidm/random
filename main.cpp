// To compile: make
// To run: or: make run
//         or: ./GreedySearch.out data_filename n
//
#define _USE_MATH_DEFINES 
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <map>
#include <vector>
#include <cmath>       /* tgamma */
#include <random>

#include <ctime> // for chrono
#include <ratio> // for chrono
#include <chrono> // for chrono

using namespace std;
using namespace std::chrono;

/*****************************************************************************************/
/*************************   CONSTANT VARIABLES  *****************************************/
/*****************************************************************************************/
#include "data.h"

/*****************************************************************************************/
/****************   GREEDY SEARCH:   Useful functions and routines    ********************/
/*****************************************************************************************/
// **** Find the best MCM, Greedy Search:
map<unsigned int, __int128_t> MCM_GreedySearch(vector<pair<__int128_t, unsigned int>> Kset, unsigned int N, unsigned int r, bool print_it = false);
map<unsigned int, __int128_t> MCM_GreedySearch_AND_printInfo(vector<pair<__int128_t, unsigned int>> Kset, unsigned int N, unsigned int r, bool print_it = false);

// **** Find the best MCM, Greedy Search starting from the model MCM_0:
map<unsigned int, __int128_t> MCM_GreedySearch_MCM0(vector<pair<__int128_t, unsigned int>> Kset, unsigned int N, unsigned int r, map<unsigned int, __int128_t> MCM_0, bool print_it = false);

// *** Greedy Search on Reduced dataset:
map<unsigned int, __int128_t> MCM_ReducedGreedySearch_AND_PrintInfo(vector<pair<__int128_t, unsigned int>> Kset, unsigned int K, unsigned int N, unsigned int r, bool print_it = false);

// *** Compare two MCMs:
void compare_two_MCMs_AND_printInfo(vector<pair<__int128_t, unsigned int>> Kset, unsigned int N, unsigned int r, map<unsigned int, __int128_t> fp1, map<unsigned int, __int128_t> fp2);


/*****************************************************************************************/
/*****************************   IMPORT an MCM from a FILE   *****************************/
/*****************************************************************************************/
// *** Read MCM from a file:
map<unsigned int, __int128_t> read_MCM_fromfile(string Input_MCM_file, unsigned int r);
map<unsigned int, __int128_t> read_MCM_fromfile_AND_printInfo(vector<pair<__int128_t, unsigned int>> Kset, unsigned int N, string Input_MCM_file, unsigned int r);


void Print_MCM_File(map<unsigned int, __int128_t> partition, unsigned int r, string extension = "_MCM");

/******************************************************************************/
/***************************   ADD OUTPUT FOLDER    ***************************/
/******************************************************************************/

//// ** location of the output folder:
string OutputFile_Add_Location(string filename)
{
    return (OUTPUT_directory + filename);
}

/****************************************************************************************************************************************************************************/
/****************************************************************************************************************************************************************************/
/**************************************************************************   """ TUTORIAL  """    **************************************************************************/
/****************************************************************************************************************************************************************************/
/****************************************************************************************************************************************************************************/

void tutorial(vector<pair<__int128_t, unsigned int>> Nset, unsigned int N,  unsigned int n)
{
    cout << endl << "*******************************************************************************************";  
    cout << endl << "******************************  CHOICE OF THE BASIS:  *************************************";
    cout << endl << "*******************************************************************************************" << endl;

// original basis of the data: this is the most natural choice a priori:
//    list<__int128_t> Basis_li = Original_Basis(n);

//// *** The basis can also be read from a file: Ex. the following files contain the best basis for the SCOTUS dataset:
   list<__int128_t> Basis_li = Read_BasisOp_IntegerRepresentation(basis_IntegerRepresentation_filename);
//   list<__int128_t> Basis_li = Read_BasisOp_BinaryRepresentation(n, basis_BinaryRepresentation_filename);

    PrintTerm_Basis(Basis_li, n);

    cout << endl << "*******************************************************************************************";
    cout << endl << "**********************  TRANSFORM the DATA in the CHOSEN BASIS   **************************";
    cout << endl << "**********************************   Build Kset:   ****************************************";
    cout << endl << "*******************************************************************************************" << endl;

//// *** Transform the data in the specified in Basis_SCModel[];

    vector<pair<__int128_t, unsigned int>> Kset = build_Kset(Nset, Basis_li);


    cout << endl << "*******************************************************************************************";
    cout << endl << "***********************  HIERARCHICAL GREEDY MERGING: BY STEPS:  **************************";
    cout << endl << "*******************************************************************************************" << endl;

    bool print_checkpoint = true;  

//// *** Finds the best MCM:
    map<unsigned int, __int128_t> fp1 = MCM_GreedySearch(Kset, N, n, print_checkpoint);

//// *** Print Log-Evidence:  
    double LogE_fp1 = LogE_MCM(Kset, fp1, N, n);
    cout << "Log-Evidence(MCM) = " << LogE_fp1 << "\t = " << LogE_fp1/((double) N)/log(2.) << " bits per datapoint \t" << endl;

//// *** Print max-Log-Likelihood:  
    double LogL_fp1 = LogL_MCM(Kset, fp1, N, n);
    cout << "Max Log-Likelihood(MCM) = " << LogL_fp1 << "\t = " << LogL_fp1/((double) N)/log(2.) << " bits per datapoint \t" << endl;

    cout << endl << "*******************************************************************************************";
    cout << endl << "******************************  READ an MCM from a FILE   *********************************";
    cout << endl << "*******************************************************************************************" << endl;

    cout << "#########  EX. READ a CHOSEN MCM:  #########" << endl;

    // the file communityfile = "INPUT/SCOTUS_Communities_inBestBasis.dat" contains the best MCM in the best basis:
    map<unsigned int, __int128_t> fp2 = read_MCM_fromfile(communityfile, n);
    Print_MCM_Partition(fp2, n);

    cout << endl << "*******************************************************************************************";
    cout << endl << "*******************************  COMPARING TWO MCMs   *************************************";
    cout << endl << "*******************************************************************************************" << endl;

    compare_two_MCMs_AND_printInfo(Kset, N, n, fp1, fp2);

    cout << endl << "*******************************************************************************************";
    cout << endl << "***************************  Decomposition of Log-E   *************************************";
    cout << endl << "*******************************   over each ICC   *****************************************";
    cout << endl << "*******************************************************************************************" << endl;

    double LogE_final = LogE_MCM_infoICC(Kset, fp1, N, n);
    //cout << "Log-Evidence(MCM) = " << LogE_final << "\t = " << LogE_final/((double) N)/log(2.) << " bits per datapoint \t" << endl;

    cout << endl << "*******************************************************************************************";
    cout << endl << "*************************  Decomposition of Max-Log-L   ***********************************";
    cout << endl << "*******************************   over each ICC   *****************************************";
    cout << endl << "*******************************************************************************************" << endl;

    double LogL_final = LogL_MCM_infoICC(Kset, fp1, N, n);
    //cout << "Max-Log-Likelihood(MCM) = " << LogL_final << "\t = " << LogL_final/((double) N)/log(2.) << " bits per datapoint \t" << endl;

    cout << endl << "*******************************************************************************************";
    cout << endl << "***************************  Working with a Reduced Dataset   *****************************";
    cout << endl << "**********   Remove from Kset all the states that occur less than K times:   **************";
    cout << endl << "*******************************************************************************************" << endl;

    // All the states that occur less than K times will be removed from the dataset:
    unsigned int K=2;
    map<unsigned int, __int128_t> fp_reduced = MCM_ReducedGreedySearch_AND_PrintInfo(Kset, K, N, n);

    cout << endl << "*******************************************************************************************";
    cout << endl << "**********************  Print information about the found MCM:  ***************************";
    cout << endl << "*******************************************************************************************" << endl;

    // Prints 1) information about the MCM; 2) the state probabilities P(s) of observed states (in the Data VS MCM); 3) the probability P(k) of observing a state with k values "+1" (in the Data VS MCM) 
    PrintFile_StateProbabilites_OriginalBasis(Nset, Basis_li, fp1, N, n, "Result");

    // Print the state probabilities P(s) of observed states (in the Data VS MCM) using the data transformed in the bew basis:
    PrintFile_StateProbabilites_NewBasis(Kset, fp1, N, n, "Result");
}

/*****************************************************************************************/
/**************************************   PARSING    *************************************/
/*****************************************************************************************/
bool new_basis = false, MCM_compare = false;

void parse_ErrorMessage()
{
    cout << endl;
    cout << "## HELP ##  Run Arguments and Options:" << endl;
    cout << "## The number of arguments must be either 0 (for default example), or at least 2:" << endl;
    cout << "\t Arg 1. full path to datafile (from current folder)" << endl;
    cout << "\t Arg 2. number of binary variables" << endl << endl;

    cout << "## Optional:" << endl;
    cout << "\t -I  followed by path to the folder containing all the input files." << endl;
    cout << "\t \t > by default this is the \'INPUT\' folder." << endl;
    cout << "\t -b  followed by name the file containing the new basis:" << endl; //"full path to file containing the new basis:" << endl;
    cout << "\t \t > basis should be written using a boolean representation (see example file 'INPUT/SCOTUS_n9_BestBasis_Binary.dat')" << endl;
    cout << "\t \t > if not specified, the defaul basis will be the original basis of the data." << endl;
    cout << "\t -m  followed by name of the file containing an MCM." << endl; //full path to file containing an MCM." << endl;
    cout << "\t \t > the MCM must be written using a boolean representation (see example file 'TO SPEFICY')." << endl;
    cout << "\t \t > the chosen MCM will be compared with the found Greedy MCM at the end of the program." << endl;
    cout << "\t -o  followed by name of the output folder." << endl;
    cout << "\t \t > this folder will be placed inside the folder \'OUTPUT\'." << endl;
    cout << "\t \t > by default all the output files will be placed directly in the \'OUTPUT\' directory." << endl;
    cout << endl;
} 

bool parse_arg(int argc, char *argv[])
{
    //// *** Read the arguments:
    if (argc == 1)
    {
        cout << "--> No arguments: Run the default example (see 'data.h' file)" << endl  << endl;
        return 1;
    }
    else if (argc == 2)
    {
        parse_ErrorMessage();
        return 0;
    }
    else //if (argc >= 3) // Two first arg must be: 'datafilename' and 'n'
    {
        string INPUT_folder = "INPUT";

        datafilename = argv[1];
        n = stoul(argv[2]);

        cout << "--> Read optional arguments (unclear arguments are ignored)" << endl;
        for(int i = 3; i < argc; i++)
        {
            if (string(argv[i]) == "-I") // new INPUT folder
            { 
                i++;
                if(i < argc) {
                    INPUT_folder = string(argv[i]);
                    cout << "Option: Input files are located in the folder: " << INPUT_folder << endl; 
                }
            }

            else if (string(argv[i]) == "-b") // Basis
            { 
                i++;
                if(i < argc) {
                    basis_BinaryRepresentation_filename = string(argv[i]);
                    new_basis = true;
                    cout << "Option: Basis (in binary representation) in file: " << basis_BinaryRepresentation_filename << endl; 
                }
            }

            else if (string(argv[i]) == "-m") // MCM to compare with
            {
                i++;
                if(i < argc) {
                    communityfile = string(argv[i]);
                    MCM_compare = true;
                    cout << "Option: MCM choice (for comparison) in file: " << communityfile << endl;
                }
            }

            else if (string(argv[i]) == "-o") // Name of the output folder (will be placed inside the 'OUTPUT' directory)
            {
                i++;
                if(i < argc) {
                    OUTPUT_directory += (string(argv[i]) + "/");
                    cout << "Option: Create new OUTPUT directory: " << OUTPUT_directory << endl;
                    system(("mkdir -p " + OUTPUT_directory).c_str());
                }
            }
            else
            {
                parse_ErrorMessage(); 
                return 0;
            }
        }

        // Add path to all files:
        datafilename = INPUT_folder + "/" + datafilename;
        basis_BinaryRepresentation_filename = INPUT_folder + "/" + basis_BinaryRepresentation_filename; 
        communityfile = INPUT_folder + "/" + communityfile;

        return 1;
    }
}

/****************************************************************************************************************************************************************************/
/****************************************************************************************************************************************************************************/
/***********************************************************************     RAND MCM FUNCTION      *************************************************************************/
/****************************************************************************************************************************************************************************/
/****************************************************************************************************************************************************************************/
//default_random_engine generator;  --> not faster than mt19937
std::mt19937 generator_mt;       //only for mt19937 

void initialise_generator()
{
          int seed = (unsigned)time(NULL);
          //srand48(seed);      //for drand48
          generator_mt.seed(seed);     //for mt19937
}

//uniform_int_distribution<unsigned int> Uniform(0,AA-1);

// Random MCM with a fixed number of parts:
// r = number of variables
// A = number of parts
map<unsigned int, __int128_t> Random_MCM_Afixed(unsigned int A, unsigned int r) //, mt19937 gen)
{
    map<unsigned int, __int128_t> MCM_rand;
    uniform_int_distribution<unsigned int> Uniform(0,A-1);
    unsigned int part;
    //int trials = 0;

    while(MCM_rand.size() != A)
    {
        MCM_rand.clear();
        __int128_t si = 1;

        for(unsigned int i=0; i<r; i++)
        {
            part = Uniform(generator_mt);
            MCM_rand[part] += si;
            si = (si << 1);
        }
        //trials++;
    }
    //cout << "number of trials = " << trials << endl;

    return MCM_rand;
}

/*
map<unsigned int, __int128_t> Random_MCM_Afixed_4(unsigned int A, unsigned int r) //, mt19937 gen)
{
    //vector<__int128_t> MCM_rand(A);
//    __int128_t MCM_rand[A] = {0};
    __int128_t* MCM_rand = (__int128_t*)malloc(A * sizeof(__int128_t));
    uniform_int_distribution<unsigned int> Uniform(0,A-1);

    unsigned int i=0;
    __int128_t si = 1;    
    unsigned int part;
    //int trials = 0;

    bool full = false;
    while(!full)
    {
        //MCM_rand.clear();
        memset(MCM_rand, 0, A * sizeof(__int128_t));
        si = 1;

        for(i=0; i<r; i++)
        {
            part = Uniform(generator_mt);
            MCM_rand[part] += si;
            si = (si << 1);
        }
        //trials++;
        full = true;
        for(i=0; i<A; i++)
        {
            if(MCM_rand[i] == 0) { full = false; break;}
        }
    }
    //cout << "number of trials = " << trials << endl;

    map<unsigned int, __int128_t> MCM_rand_map;
    for(i=0; i<A; i++)
        {  MCM_rand_map[i] = MCM_rand[i]; }

    return MCM_rand_map;
}

map<unsigned int, __int128_t> Random_MCM_Afixed_2(unsigned int A, unsigned int r) //, mt19937 gen)
{
    vector<__int128_t> MCM_rand(A);
    uniform_int_distribution<unsigned int> Uniform(0,A-1);

    unsigned int i=0;
    __int128_t si = 1;    
    unsigned int part;
    //int trials = 0;
    bool full = false;

    while(!full)  //(MCM_rand.size() != A)
    {
        fill(MCM_rand.begin(), MCM_rand.end(), 0);
        si = 1;

        for(i=0; i<r; i++)
        {
            part = Uniform(generator_mt);
            MCM_rand[part] += si;
            si = (si << 1);
        }
        full = true;
        for(i=0; i<A; i++)
        {
            if(MCM_rand[i] == 0) { full = false; break;}
        }
        //trials++;
    }
    //cout << "number of trials = " << trials << endl;

    map<unsigned int, __int128_t> MCM_rand_map;
    i=0;
    for(auto& ICC : MCM_rand)   //i=0; i<A; i++)
    {  
        MCM_rand_map[i] = ICC; 
        i++;
    }

    return MCM_rand_map;
}*/

double Random_MCM_Afixed_histo(vector<pair<__int128_t, unsigned int>> Kset, unsigned int N, unsigned int r, unsigned int A, unsigned int Nit = 10000, string datafilename = "MCM_rand")
{
    auto start = chrono::system_clock::now();

    fstream file_MCM_rand(OutputFile_Add_Location(datafilename + ".dat"), ios::out);
    file_MCM_rand << "## Nit = " << Nit << " Random models with A = " << A << " partitions (ICCS) over n = " << n << " variables:" << endl; 
    file_MCM_rand << "## " << endl; 
    file_MCM_rand << "## 1:LogE \t 2:LogE in bits/datapoint" << endl; 

    double LogE=0;
    double LogE_best= -((double) N) * r * log(2.);
    double divisor = ((double) N) * log(2.);

    for (int i=0; i< Nit; i++)
    {
        LogE = LogE_MCM(Kset, Random_MCM_Afixed(A, r), N, r);
        file_MCM_rand << LogE << " \t " << LogE / divisor << endl;
        if(LogE > LogE_best) {  LogE_best = LogE; }
    }

    file_MCM_rand.close();

    auto end = chrono::system_clock::now();

    // *** Time it takes to find partition
    chrono::duration<double> elapsed = end - start;
    cout << "Run time Random_MCM function : " << elapsed.count() << "s" << endl << endl;

    return LogE_best;
}

/****************************************************************************************************************************************************************************/
/****************************************************************************************************************************************************************************/
/**************************************************************************     MAIN FUNCTION      **************************************************************************/
/****************************************************************************************************************************************************************************/
/****************************************************************************************************************************************************************************/

int main(int argc, char *argv[])
{
    cout << endl << "*******************************************************************************************";
    cout << endl << "*****************************  INITIALISATION OF THE PROGRAM:  ****************************";
    cout << endl << "*******************************************************************************************" << endl;

    cout << "--->> Create the \"OUTPUT\" Folder: (if needed) ";
    system(("mkdir -p " + OUTPUT_directory).c_str());
    cout << endl;

    if (!parse_arg(argc, argv)) { return 0; }; // if there is an issue with the parsing --> quit

    cout << endl << "*******************************************************************************************";
    cout << endl << "***********************************  READ THE DATA:  **************************************";
    cout << endl << "*******************************************************************************************" << endl;

    unsigned int N = 0; // will contain the number of datapoints in the dataset
    vector<pair<__int128_t, unsigned int>> Nset = read_datafile(&N, datafilename, n);

    if (N == 0)  { return 0; }  // Terminate program if the file can't be found or read, or if it is empty:

    cout << endl << "*******************************************************************************************";  
    cout << endl << "******************************  CHOICE OF THE BASIS:  *************************************";
    cout << endl << "*******************************************************************************************" << endl;

    vector<pair<__int128_t, unsigned int>> Kset;
    list<__int128_t> Basis_li;

    if (new_basis) {
        Basis_li = Read_BasisOp_BinaryRepresentation(n, basis_BinaryRepresentation_filename); // Read basis in binary representation from file
        //Basis_li = Read_BasisOp_IntegerRepresentation(basis_IntegerRepresentation_filename);   // Read basis in integer representation from file

        if (Basis_li.size() != 0) {
            cout << endl << "## Chosen basis in file: " << basis_BinaryRepresentation_filename << endl;
            PrintTerm_Basis(Basis_li, n);

            cout << endl << "## Transform the data in the new basis: ";
            Kset = build_Kset(Nset, Basis_li);
        }
        else {
//            cout << endl << "## Unable to read the basis: " << endl;
            cout << endl << "## Data is kept in the original basis. " << endl;
            Basis_li = Original_Basis(n);   // original basis of the data: this is the most natural choice a priori
            Kset = Nset;
        }
    }
    else {
        cout << endl << "## Chosen basis is the original basis of the data: the data is not modified." << endl;
        Basis_li = Original_Basis(n);   // original basis of the data: this is the most natural choice a priori
        Kset = Nset;
    }
/*
    cout << endl << "*******************************************************************************************";
    cout << endl << "*****************************  HIERARCHICAL GREEDY MERGING:  ******************************";
    cout << endl << "********************************  in the CHOSEN BASIS  ************************************";
    cout << endl << "*******************************************************************************************" << endl;

    bool print_checkpoint = false;  

//// *** Finds the best MCM and print information about it in the terminal:
    map<unsigned int, __int128_t> mcm1 = MCM_GreedySearch_AND_printInfo(Kset, N, n, print_checkpoint);

    Print_MCM_File(mcm1, n, "MCM_Best");

    cout << endl << "*******************************************************************************************";
    cout << endl << "**********************  Decomposition of Log-E and Log-L   ********************************";
    cout << endl << "*******************************   over each ICC   *****************************************";
    cout << endl << "*******************************************************************************************" << endl;

    cout << endl << "######### Information Log-Evidence: #########" << endl;
    LogE_MCM_infoICC(Kset, mcm1, N, n);

    cout << endl << "######### Information Log-Likelihood: #######" << endl;
    LogL_MCM_infoICC(Kset, mcm1, N, n);

    if(MCM_compare) {
    cout << endl << "*******************************************************************************************";
    cout << endl << "*******************************  COMPARING TWO MCMs   *************************************";
    cout << endl << "*******************************************************************************************" << endl;
        map<unsigned int, __int128_t> mcm2 = read_MCM_fromfile(communityfile, n);
        //Print_MCM_Partition(mcm2, n);
        compare_two_MCMs_AND_printInfo(Kset, N, n, mcm1, mcm2);
    }
*/

    cout << endl << "*******************************************************************************************";
    cout << endl << "*************************************  RANDOM MCM   ***************************************";
    cout << endl << "*******************************************************************************************" << endl;
//    initialise_generator();

    unsigned int A = 5;
    unsigned int Nit = 100;

    map<unsigned int, __int128_t> MCM_rand = Random_MCM_Afixed(A, n); 
    Print_MCM_Partition(MCM_rand, n);
    LogE_MCM_infoICC(Kset, MCM_rand, N, n);
    
    // Already tested in comparison with having:
    //   -- declaration of uniform distribution in/out of function "Random_MCM_Afixed()"
    //   -- using "vector" or "array" instead of "map" --> This is not faster, at least in cases where A is small.

//    double LogE_rand_best = Random_MCM_Afixed_histo(Kset, N, n, A, Nit, "MCM_rand_B0");
    double LogE_rand_best = Random_MCM_Afixed_histo(Kset, N, n, A, Nit, "MCM_rand_Bbest");

    cout << "Best LogE among " << Nit << " random MCMs with A = " << A << " ICCs is: LogE(rand) = " << LogE_rand_best;
    cout << " = " << LogE_rand_best/((double) N)/log(2.) << " bits per datapoint" << endl << endl;

/*
    cout << endl << "*******************************************************************************************************************";
    cout << endl << "*******************************************************************************************************************";
    cout << endl << "************************************************  TUTORIAL:  ******************************************************";
    cout << endl << "*******************************************************************************************************************";
    cout << endl << "*******************************************************************************************************************" << endl;

    tutorial(Nset, N, n);
*/

    return 0;
}
