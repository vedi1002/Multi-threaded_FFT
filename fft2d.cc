// Distributed two-dimensional Discrete FFT transform
// Pranshu Trivedi

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>

#include "Complex.h"
#include "InputImage.h"
#define _USE_MATH_DEFINES
#include <cmath>

constexpr unsigned int NUMTHREADS = 4;

using namespace std;

//undergrad students can assume NUMTHREADS will evenly divide the number of rows in tested images
//graduate students should assume NUMTHREADS will not always evenly divide the number of rows in tested images.
// I will test with a different image than the one given

Complex func_e(double, double, int);
Complex inversefunc_e(double, double, int);
void Transform1D(Complex*, int, Complex*, bool);
void thread_func(std::vector<Complex*> rowColPtrs, int w, std::vector<Complex*> outPtrs, bool isRow, int index, int size);
void InverseTransform1D(Complex* h, int w, Complex* H, bool isRow);
void in_thread_func(std::vector<Complex*> rowColPtrs, int w, std::vector<Complex*> outPtrs, bool isRow, int index, int size);

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Create a vector of complex objects of size width * height to hold
  //    values calculated
  // 3) Do the individual 1D transforms on the rows assigned to each thread
  // 4) Force each thread to wait until all threads have completed their row calculations
  //    prior to starting column calculations
  // 5) Perform column calculations
  // 6) Wait for all column calculations to complete
  // 7) Use SaveImageData() to output the final results

  InputImage image(inputFN);  // Create the helper object for reading the image
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-7
  int w = image.GetHeight();
  std::vector<Complex> container(w * w);
  int rowsPerThread = w / NUMTHREADS;

  Complex* inputData = image.GetImageData();

  std::vector<Complex*> rowPtrs(w);
  std::vector<Complex*> outPtrs(w);

  for (int i = 0; i < w; i++) {
    rowPtrs[i] = inputData + (i*w);
    outPtrs[i] = &(*container.begin()) + (i*w);
  }

  std::thread threadArr[NUMTHREADS];

  int counter = 0;
  for (int i = 0; i < w; i += rowsPerThread) {
    threadArr[counter] = std::thread(thread_func, rowPtrs, w, outPtrs, true, i, rowsPerThread);
    counter++;
  }

  for (int i = 0; i < NUMTHREADS; i++) threadArr[i].join();

  //columns
  std::vector<Complex*> colPtrs(w);
  std::vector<Complex*> outColPtrs(w);
  std::vector<Complex> finalContainer(w*w);

  for (int i=0; i < w; i++){
    colPtrs[i] = &(*container.begin()) + i;
    outColPtrs[i] = &(*finalContainer.begin()) + i;
  }

  std::thread newThreadArr[NUMTHREADS];
  counter = 0;
  for (int i = 0; i < w; i += rowsPerThread) {
    newThreadArr[counter] = std::thread(thread_func, colPtrs, w, outColPtrs, false, i, rowsPerThread);
    counter++;
  }

  for (int i=0; i < NUMTHREADS; i++) newThreadArr[i].join();

  char filename[14] = "MyAfter2D.txt";

  image.SaveImageData(filename, &(*finalContainer.begin()), w, w);


  //INVERSE
  std::vector<Complex> inr_container(w*w);
  std::vector<Complex*> in_rowPtrs(w);
  std::vector<Complex*> in_outPtrs(w);

  for (int i = 0; i < w; i++) {
    in_rowPtrs[i] = &(*finalContainer.begin()) + (i*w);
    in_outPtrs[i] = &(*inr_container.begin()) + (i*w);
  }

  std::thread in_threadArr[NUMTHREADS];

  counter = 0;
  for (int i = 0; i < w; i += rowsPerThread) {
    in_threadArr[counter] = std::thread(in_thread_func, in_rowPtrs, w, in_outPtrs, true, i, rowsPerThread);
    counter++;
  }

  for (int i = 0; i < NUMTHREADS; i++) in_threadArr[i].join();

  //columns
  std::vector<Complex*> in_colPtrs(w);
  std::vector<Complex*> in_outColPtrs(w);
  std::vector<Complex> in_finalContainer(w*w);

  for (int i=0; i < w; i++){
    in_colPtrs[i] = &(*inr_container.begin()) + i;
    in_outColPtrs[i] = &(*in_finalContainer.begin()) + i;
  }

  std::thread in_newThreadArr[NUMTHREADS];
  counter = 0;
  for (int i = 0; i < w; i += rowsPerThread) {
    in_newThreadArr[counter] = std::thread(in_thread_func, in_colPtrs, w, in_outColPtrs, false, i, rowsPerThread);
    counter++;
  }

  for (int i=0; i < NUMTHREADS; i++) in_newThreadArr[i].join();

  char in_filename[19] = "MyAfterInverse.txt";

  image.SaveImageDataReal(in_filename, &(*in_finalContainer.begin()), w, w);



}





//handles the given number of rows/columns
void thread_func(std::vector<Complex*> rowColPtrs, int w, std::vector<Complex*> outPtrs, bool isRow, int index, int size) {

  for (int i = index; i < index + size; i++)
    Transform1D(rowColPtrs[i], w, outPtrs[i], isRow);

}

void in_thread_func(std::vector<Complex*> rowColPtrs, int w, std::vector<Complex*> outPtrs, bool isRow, int index, int size) {

  for (int i = index; i < index + size; i++)
    InverseTransform1D(rowColPtrs[i], w, outPtrs[i], isRow);

}

void Transform1D(Complex* h, int w, Complex* H, bool isRow)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  Complex sum;
  int innerCounter = 0;
  int outerCounter = 0;

  //iterate through H
  for (int k = 0; k < w; k++) {
    sum = Complex(0,0);
    innerCounter = 0;

    //iterate through h
    for (int n = 0; n < w; n++) {
      sum = sum + (*(h + innerCounter) * func_e(k,n,w));

      if (isRow)
        innerCounter += 1;
      else
        innerCounter += w;
    }

    *(H+outerCounter) = sum;

    if (isRow)
      outerCounter += 1;
    else
      outerCounter += w;
  }
}

void InverseTransform1D(Complex* h, int w, Complex* H, bool isRow)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  Complex sum;
  int innerCounter = 0;
  int outerCounter = 0;

  //iterate through H
  for (int k = 0; k < w; k++) {
    sum = Complex(0,0);
    innerCounter = 0;

    //iterate through h
    for (int n = 0; n < w; n++) {
      sum = sum + (*(h + innerCounter) * inversefunc_e(k,n,w));

      if (isRow)
        innerCounter += 1;
      else
        innerCounter += w;
    }

    *(H+outerCounter) = Complex(sum.real * (1/ (double) w), sum.imag * (1/ (double) w));

    if (isRow)
      outerCounter += 1;
    else
      outerCounter += w;
  }
}


Complex func_e(double k, double n, int w) {
  double in = (2.0 * M_PI * k * n) / w;
  return Complex(cos(in), -1.0 * sin(in));
}

Complex inversefunc_e(double k, double n, int w) {
  double in = (2.0 * M_PI * k * n) / w;
  return Complex(cos(in), sin(in));
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  Transform2D(fn.c_str()); // Perform the transform.
}  
  

  
