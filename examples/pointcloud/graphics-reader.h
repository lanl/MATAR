/**********************************************************************************************
 © 2020. Triad National Security, LLC. All rights reserved.
 This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
 National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
 Department of Energy/National Nuclear Security Administration. All rights in the program are
 reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
 Security Administration. The Government is granted for itself and others acting on its behalf a
 nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
 derivative works, distribute copies to the public, perform publicly and display publicly, and
 to permit others to do so.
 This program is open source under the BSD-3 License.
 Redistribution and use in source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:
 
 1.  Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.
 
 2.  Redistributions in binary form must reproduce the above copyright notice, this list of
 conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.
 
 3.  Neither the name of the copyright holder nor the names of its contributors may be used
 to endorse or promote products derived from this software without specific prior
 written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************/
 
// -----------------------------------------------
// routines to read a graphics file
//
// -----------------------------------------------

#include <iostream>
#include <stdio.h>
#include <sys/stat.h>
#include <fstream>
#include <cmath>
#include <vector>

#include "matar.h"

using namespace mtr;



// checks to see if a path exists
bool DoesPathExist(const std::string &s)
{
    struct stat buffer;
    return (stat (s.c_str(), &buffer) == 0);
}

// Code from stackover flow for string delimiter parsing
std::vector<std::string> split (std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
    
} // end of split


// retrieves multiple values between [ ]
std::vector<double> extract_list(std::string str) {
    
    // replace '[' with a space and ']' with a space
    std::replace(str.begin(), str.end(), '[', ' ');
    std::replace(str.begin(), str.end(), ']', ' ');
    
    std::vector<std::string> str_values;
    std::vector<double> values;

    // exact the str values into a vector
    str_values = split(str, ",");
    
    // convert the text values into double values
    for (auto &word : str_values) {
        values.push_back( atof(word.c_str()) );
    } // end for
    
    return values;
    
}  // end of extract_list



// from stack overflow on removing blanks in string
template<typename T, typename P>
T remove_if(T beg, T end, P pred)
{
    T dest = beg;
    for (T itr = beg;itr != end; ++itr)
        if (!pred(*itr))
            *(dest++) = *itr;
    return dest;
}







// This function reads a VTK file
//--------------------------------------------------------
//
/*
 vtk mesh node ordering
          7--------6
         /|       /|
        / |      / |
       4--------5  |
       |  |     |  |
       |  |     |  |
       |  3-----|--2
       | /      | /
       |/       |/
       0--------1
 
 in the i,j,k format node order is
 0 = (i  , j  , k  )
 1 = (i+1, j  , k  )
 2 = (i+1, j+1, k  )
 3 = (i  , j+1, k  )
 4 = (i  , j  , k+1)
 5 = (i+1, j  , k+1)
 6 = (i+1, j+1, k+1)
 7 = (i  , j+1, k+1)
 
 The marching cubes ordering is
 (i  , j  , k  ) = 0
 (i+1, j  , k  ) = 1
 (i+1, j  , k+1) = 5
 (i  , j  , k+1) = 4
 (i  , j+1, k  ) = 3
 (i+1, j+1, k  ) = 2
 (i+1, j+1, k+1) = 6
 (i  , j+1, k+1) = 7

 */
void readVTK(char * filename,
             CArray <double> &pt_coords,
             CArray <int> &elem_point_list,
             CArray <double> &pt_values,
             int &num_points,
             int &num_elems,
             int &num_points_in_elem,
             bool test_vtk_read,
             bool verbose_vtk)
{
   
    int i;           // used for writing information to file
    int point_id;    // the global id for the point
    int elem_id;     // the global id for the elem
    int this_point;   // a local id for a point in a elem (0:7 for a Hexahedral elem)
    
    int num_dims = 3;
    

    std::string token;
    
    bool found = false;
    
    std::ifstream in;  // FILE *in;
    in.open(filename);
    

    // look for POINTS
    i = 0;
    while (found==false) {
        std::string str;
        std::string delimiter = " ";
        std::getline(in, str);
        std::vector<std::string> v = split (str, delimiter);
        
        // looking for the following text:
        //      POINTS %d float
        if(v[0] == "POINTS"){
            num_points = std::stoi(v[1]);
            printf("Num nodes read in %d\n", num_points);
            
            found=true;
        } // end if
        
        
        if (i>1000){
            printf("ERROR: Failed to find POINTS \n");
            break;
        } // end if
        
        i++;
    } // end while
    
    
    // allocate memory for point coords and values
    pt_coords = CArray <double> (num_points,3);
    pt_values = CArray <double> (num_points);
    
    
    // read the point coordinates
    for (point_id=0; point_id<num_points; point_id++){
        
        std::string str;
        std::getline(in, str);
        
        std::string delimiter = " ";
        std::vector<std::string> v = split (str, delimiter);
        
        for (int dim=0; dim<3; dim++){
            pt_coords(point_id,dim) = std::stod(v[dim]); // double
            //if num_dims=2 skip the 3rd value
            
            // printing all the mesh coordinates
            if (verbose_vtk) printf(" %f ", pt_coords(point_id,dim));
        }
        if (verbose_vtk) printf("\n"); // printing a space for readability
        
    } // end for points
    found=false;
    
    
    if (verbose_vtk)printf("\n");
    if (verbose_vtk) printf("looking for CELLS \n");
    
    // look for CELLS
    i = 0;
    while (found==false) {
        std::string str;
        std::getline(in, str);
        
        std::string delimiter = " ";
        std::vector<std::string> v = split (str, delimiter);
        
        // looking for the following text:
        //      CELLS num_elems size
        if(v[0] == "CELLS"){
            num_elems = std::stoi(v[1]);
            printf("Num elements read in %d\n", num_elems);
            
            found=true;
        } // end if
        
        
        if (i>1000){
            printf("ERROR: Failed to find CELLS \n");
            break;
        } // end if
        
        i++;
    } // end while
    
    
    
    // allocate memomry for points in each element
    elem_point_list = CArray <int> (num_elems,8);   // 8 points in a hex
    
    
    // read the point ids in the element
    for (elem_id=0; elem_id<num_elems; elem_id++) {
        
        std::string str;
        std::getline(in, str);
        
        std::string delimiter = " ";
        std::vector<std::string> v = split (str, delimiter);
        num_points_in_elem = std::stoi(v[0]);
        
        for (this_point=0; this_point<num_points_in_elem; this_point++){
            elem_point_list(elem_id,this_point) = std::stod(v[this_point+1]);
            
            // printing details on nodes in the element
            if (verbose_vtk) printf(" %d ", elem_point_list(elem_id,this_point) );
        }
        if (verbose_vtk) printf("\n"); // printing a space for readability
        
    } // end for
    found=false;

    if (verbose_vtk) printf("\n"); // printing a space for readability
    
    
    // look for CELL_TYPE
    i = 0;
    int elem_type = 0;
    while (found==false) {
        std::string str;
        std::string delimiter = " ";
        std::getline(in, str);
        std::vector<std::string> v = split (str, delimiter);
        
        // looking for the following text:
        //      CELLS num_elems size
        if(v[0] == "CELL_TYPES"){

            std::getline(in, str);
            elem_type = std::stoi(str);
            
            found=true;
        } // end if
        
        
        if (i>1000){
            printf("ERROR: Failed to find elem_TYPE \n");
            break;
        } // end if
        
        i++;
    } // end while
    
    if (verbose_vtk) printf("elem type = %d \n", elem_type);
    // elem types:
    // linear hex = 12, linear quad = 9
    found=false;
    
    // verify mesh has hexahedral elements, which have 8 nodes
    if(num_points_in_elem != 8) {
        printf("wrong elem type of %d \n", elem_type);
    }
    
    
    
    // look for the point_var in the POINT_DATA heading
    i = 0;
    while (found==false) {
        std::string str;
        std::string delimiter = " ";
        std::getline(in, str);
        std::vector<std::string> v = split (str, delimiter);
        
        // looking for the following text:
        //      POINT_DATA num_points
        if(v[1] == "point_var"){
            
            std::getline(in, str);  // read next line -- its LOOKUP_TABLE
            
            //
            for(int point_id=0; point_id<num_points; point_id++){
                std::getline(in, str);
                pt_values(point_id) = std::stoi(str);
                
                
                // printing the node values in the mesh
                if (verbose_vtk) printf("%f \n", pt_values(point_id) );
            }
            if (verbose_vtk) printf("\n"); // printing a blank space
        
            found=true;
            
        } // end if
        
        
        if (i>10000000){
            printf("ERROR: Failed to find point_var in POINT_DATA \n");
            break;
        } // end if
        
        i++;
    } // end while


    found=false;
    
    
    
    // testing the file read by painting a part
    if(test_vtk_read == true){
        for(int node_id = 0; node_id<num_points; node_id++){
            
            double x = pt_coords(node_id,0);
            double y = pt_coords(node_id,1);
            double z = pt_coords(node_id,2);
            
            // a simple sphere
            pt_values(node_id) = sqrt(x*x + y*y + z*z) - 0.5;
        }
    } // end of test=true
    
    
    printf("Finished reading mesh \n\n");
    
    in.close();
    
}


// an array to convert the marchint cubes id order to marching cubes id order
const int marching_cubes_2_vtk[8] =
{
    0,
    1,
    5,
    4,
    3,
    2,
    6,
    7
};



void readTechPlot(char * filename,
             CArray <double> &pt_coords,
             CArray <int> &elem_point_list,
             CArray <double> &pt_values,
             int &num_points,
             int &num_elems,
             int &num_points_in_elem,
             bool test_tecplot_read,
             bool verbose_vtk)
{
   
    int i;           // used for writing information to file
    int point_id;    // the global id for the point
    int elem_id;     // the global id for the elem
    
    int num_dims = 3;
    

    std::string token;
    
    bool found = false;
    
    std::ifstream in;  // FILE *in;
    in.open(filename);
    

    // look for POINTS
    i = 0;  // lines in the file
    while (found==false) {
        std::string str;
        std::string delimiter = ",";
        std::getline(in, str);
        std::vector<std::string> v = split (str, delimiter);
        
        bool found_nodes = false;
        bool found_elems = false;
        
        // loop over the parsed text stored in v and
        // am now looking for the following text:
        //      NODES %d float
        for (int text=0; text<v.size(); text++){

            
            std::string delimiter_words = "=";
            std::vector<std::string> words = split (v[text], delimiter_words);
            
            
            
            for(int a_word=0; a_word<words.size(); a_word++){
                
                // erase extra spaces from the text, the remaining text are names and numbers
                words[a_word].erase(std::remove_if(words[a_word].begin(), words[a_word].end(),
                    [](char c) { return std::isspace(c); } ),
                                  words[a_word].end());
                
                
                if(words[a_word] == "NODES"){
                    num_points = std::stoi(words[a_word+1]);
                    printf("Num nodes to read in %d\n", num_points);
                    
                    found_nodes = true;
                } // end if
                
                if(words[a_word] == "ELEMENTS"){
                    num_elems = std::stoi(words[a_word+1]);
                    printf("Num elements to read in %d\n", num_elems);
                    
                    found_elems = true;
                }
                
                if(found_nodes == true && found_elems == true){
                    found = true;
                }
            } // end loop over all the words in the text within a vector
            
        } // end for loop over text in the line
        
        if (i>=2) found=true;
        
        if (i>1000){
            printf("ERROR: Failed to find NODES and ELEMENTS \n");
            break;
        } // end if
        
        i++;
    } // end while
    
    
    // allocate memory for point coords and values
    pt_coords = CArray <double> (num_points,3);
    pt_values = CArray <double> (num_points);
    

    printf("starting the x,y,z point read and density value \n");
    
    // read the point coordinates
    for (point_id=0; point_id<num_points; point_id++){
        
        std::string str;
        std::getline(in, str);
        
        std::string delimiter = " ";
        std::vector<std::string> v = split (str, delimiter);
        
        int column = 0;
        for (int text=0; text<v.size(); text++){
            
            // erase extra spaces from the text, the remaining text is a number
            v[text].erase(std::remove_if(v[text].begin(), v[text].end(),
                [](char c) { return std::isspace(c); } ),
                          v[text].end());
            
            
            // column numbering starts at 0
            if(v[text].size() > 0 && column==3){
                pt_values(point_id) = std::stod(v[text]); // double
                
                if (verbose_vtk) printf("    %f \n", pt_values(point_id));
                
                break; // exit after reading the density on this line
            } // end if column is density
            
            
            // if there is text, then it is a number
            if(v[text].size() > 0 && column<3){
                
                
                pt_coords(point_id,column) = std::stod(v[text]); // double
                //if num_dims=2 skip the 3rd value
            
                // printing all the mesh coordinates
                if (verbose_vtk) printf(" %f ", pt_coords(point_id,column));
                
                column++; // this will make the column = 3 after all dims are saved
            } // end if to save coordinates
            
            
        } // end for over the partitioned test
        
    } // end for points
    found=false;
    
    

    
    // allocate memomry for points in each element
    num_points_in_elem = 8;
    elem_point_list = CArray <int> (num_elems,8);   // 8 points in a hex
    
    
    // read the point ids in the element
    if (verbose_vtk) printf("Reading the nodes in the element\n");
    for (elem_id=0; elem_id<num_elems; elem_id++) {
        
        std::string str;
        std::getline(in, str);
        
        std::string delimiter = " ";
        std::vector<std::string> v = split (str, delimiter);
        
        
        int column = 0;
        for (int text=0; text<v.size(); text++){
            
            // erase extra spaces from the text, the remaining text is a number
            v[text].erase(std::remove_if(v[text].begin(), v[text].end(),
                [](char c) { return std::isspace(c); } ),
                          v[text].end());
            
            
            
            if(column==8){
                if (verbose_vtk) printf("\n"); // printing a space for readability
                
                break;  // exit after reading the last node on this line
            }
            
            // if there is text, then it is a number
            if(v[text].size() > 0 && column<8){
                
                
                elem_point_list(elem_id,column) = std::stod(v[text]) - 1; // Fortran convention
                //if num_dims=2 skip the 3rd value
            
                // printing all the mesh coordinates
                // printing details on nodes in the element
                if (verbose_vtk) printf(" %d ", elem_point_list(elem_id,column) );
                
                column++; // this will make the column = 8 after all node values are saved
            } // end if to save points in this element
            
            
        } // end loop over the text
        
    } // end for
    found=false;

    if (verbose_vtk) printf("\n"); // printing a space for readability
 
        

    
    // testing the file read by painting a part
    if(test_tecplot_read == true){
        for(int node_id = 0; node_id<num_points; node_id++){
            
            
            pt_coords(node_id,0) *= 100;
            pt_coords(node_id,1) *= 100;
            pt_coords(node_id,2) *= 100;
            
            double x = pt_coords(node_id,0);
            double y = pt_coords(node_id,1);
            double z = pt_coords(node_id,2);
            
            // a simple sphere
            pt_values(node_id) = sqrt(x*x + y*y + z*z) - 0.5;
        }
    } // end of test=true
     

    
    printf("Finished reading mesh \n\n");
    
    in.close();
    
}





// -------------------------------------------------------
// This function write outs the data to a VTK file
//--------------------------------------------------------
//
void VTK(CArray <double> &pt_coords,
         CArray <int> &elem_point_list,
         int num_points,
         int num_elems,
         int num_points_in_elem,
         CArray <double> &pt_values)
{
    
    int GraphicsNumber = 0;
    double Time = 0.0;
    
   
    int i;           // used for writing information to file
    int point_id;    // the global id for the point
    int elem_id;     // the global id for the elem
    int this_point;   // a local id for a point in a elem (0:7 for a Hexahedral elem)
    
    
    FILE *out[20];   // the output files that are written to
    char name[100];  // char string
    
    
    
    std::string directory = "vtk";
    bool path = DoesPathExist(directory);
    
    // Create the folders for the ensight files
    if (path==false) {
        i=system("mkdir vtk");
    }
    
    
    
    
    /*
     ---------------------------------------------------------------------------
     Write the Geometry file
     ---------------------------------------------------------------------------
     */
    
    
    snprintf(name, sizeof(name), "vtk/mesh.vtk");  // mesh file
    
    
    out[0]=fopen(name,"w");
    
    
    fprintf(out[0],"# vtk DataFile Version 2.0\n");  // part 2
    fprintf(out[0],"Mesh for Fierro\n");             // part 2
    fprintf(out[0],"ASCII \n");                      // part 3
    fprintf(out[0],"DATASET UNSTRUCTURED_GRID\n\n"); // part 4
    
    fprintf(out[0],"POINTS %d float\n", num_points);

    
    // write all components of the point coordinates
    for (point_id=0; point_id<num_points; point_id++){
        fprintf(out[0],
                "%f %f %f\n",
                pt_coords(point_id,0),
                pt_coords(point_id,1),
                pt_coords(point_id,2));
    } // end for
    
    /*
     ---------------------------------------------------------------------------
     Write the elems
     ---------------------------------------------------------------------------
     */
    fprintf(out[0],"\n");
    fprintf(out[0],"CELLS %d %d\n", num_elems, num_elems+num_elems*8);  // size=all printed values
    
    // write all global point numbers for this elem
    for (elem_id=0; elem_id<num_elems; elem_id++) {
        
        fprintf(out[0],"8 "); // num points in this elem
        for (this_point=0; this_point<num_points_in_elem; this_point++){
            fprintf(out[0],"%d ", elem_point_list(elem_id,this_point));
        }
        fprintf(out[0],"\n");
        
    } // end for
    
    fprintf(out[0],"\n");
    fprintf(out[0],"CELL_TYPES %d \n", num_elems);
    // elem types:
    // linear hex = 12, linear quad = 9
    // element types: https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    // element types: https://kitware.github.io/vtk-js/api/Common_DataModel_CellTypes.html
    // vtk format: https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    for (elem_id=0; elem_id<num_elems; elem_id++) {
        fprintf(out[0],"%d \n", 12); // linear hex is type 12
    }
    
    
    /*
     ---------------------------------------------------------------------------
     Write the nodal variable file
     ---------------------------------------------------------------------------
     */
    fprintf(out[0],"\n");
    fprintf(out[0],"POINT_DATA %d \n", num_points);
    fprintf(out[0],"SCALARS point_var float 1\n"); // the 1 is number of scalar components [1:4]
    fprintf(out[0],"LOOKUP_TABLE default\n");
    for (point_id=0; point_id<num_points; point_id++) {
        double var=2;
        fprintf(out[0],"%f\n",pt_values(point_id));
    }
    
    /*
     ---------------------------------------------------------------------------
     Write the vector variables to file
     ---------------------------------------------------------------------------
     */

    
    /*
     ---------------------------------------------------------------------------
     Write the scalar elem variable to file
     ---------------------------------------------------------------------------
     */
    fprintf(out[0],"\n");
    fprintf(out[0],"CELL_DATA %d \n", num_elems);
    fprintf(out[0],"SCALARS elem_var float 1\n"); // the 1 is number of scalar components [1:4]
    fprintf(out[0],"LOOKUP_TABLE default\n");
    for (elem_id=0; elem_id<num_elems; elem_id++) {
        double var=1;
        fprintf(out[0],"%f\n",var);
    }
    
    fprintf(out[0],"\n");
    fprintf(out[0],"SCALARS elem_var2 float 1\n"); // the 1 is number of scalar components [1:4]
    fprintf(out[0],"LOOKUP_TABLE default\n");
    for (elem_id=0; elem_id<num_elems; elem_id++) {
        double var=10;
        fprintf(out[0],"%f\n",var);
    }
    
    fclose(out[0]);

}

