#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

// allows read in word by word, not line by line
string temp;

//define vectors globally
vector<string> stationName;
vector<int> yearRecorded;
vector<int> monthRecorded;
vector<int> dayRecorded;
vector<int> timeRecorded;
vector<int> airTemp;

// instantiate sizing for array, ready for SD later
int initialSize;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

// Method used to calculate data
void readData() {
	// Read data in from Text File
	std::cout << "Reading..." << std::endl;

	//std::ifstream file("temp_lincolnshire_short.txt");
	std::ifstream file("temp_lincolnshire.txt");

	// set counter to 0 to allow incrementation 
	int counter = 0;

	// initialise clock for read time
	clock_t start = clock();

	while (file >> temp) {
		switch (counter)
		{

		// use vector::push_back, Adds a new element at the end of the vector, 
		// after its current last element. 
		// src: http://www.cplusplus.com/reference/vector/vector/push_back/
		case 0: 
			stationName.push_back(temp);
			//increment counter
			counter++;
			break;
		case 1:
			// use 'stoi' to convert string to int
			// src: http://www.cplusplus.com/reference/string/stoi/
			yearRecorded.push_back(stoi(temp));
			counter++;
			break;
		case 2:
			monthRecorded.push_back(stoi(temp));
			counter++;
			break;
		case 3:
			dayRecorded.push_back(stoi(temp));
			counter++;
			break;
		case 4:
			// use 'stoi' to convert string to float
			// src: http://www.cplusplus.com/reference/string/stof/
			timeRecorded.push_back(stoi(temp));
			counter++;
			break;
		case 5:
			airTemp.push_back(stof(temp) * 10);
			counter = 0;
			break;
		}
	}

	// close file 
	file.close();
	// stop timer
	clock_t finish = clock();

	// output info
	std::cout << "\n*********************" << std::endl;
	std::cout << "File read" << std::endl;
	std::cout << "Total file read time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
	std::cout << "*********************" << std::endl;

	// set initial size to air temperature array size, for SD division later
	initialSize = airTemp.size();
}

// Method used to calculate Minimum of values
float getMinimum(cl::Context context, cl::CommandQueue queue, cl::Program program) {
	
	typedef int mytype;
	cl::Event prof_event;

	//Part 4 - memory allocation
	//host - input
	//std::vector<mytype> A(10, 1);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
	vector<int> A = airTemp;

	//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
	//if the total input length is divisible by the workgroup size
	//this makes the code more efficient
	size_t local_size = 256;

	// adjust padding size according to the length of the array modulus the local size
	size_t padding_size = A.size() % local_size;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<int> A_ext(local_size - padding_size, 0);
		//append that extra vector to our input
		A.insert(A.end(), A_ext.begin(), A_ext.end());
	}

	size_t input_elements = A.size();//number of input elements
	size_t input_size = A.size() * sizeof(mytype);//size in bytes
	size_t nr_groups = input_elements / local_size;//define number of groups

	//host - output
	std::vector<mytype> B(input_elements);
	size_t output_size = B.size() * sizeof(mytype);//size in bytes

	//device - buffers
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

	//Part 5 - device operations

	//5.1 copy array A to and initialise other arrays on device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

	//5.2 Setup and execute all kernels (i.e. device code)
	cl::Kernel kernel_1 = cl::Kernel(program, "minimum");
	kernel_1.setArg(0, buffer_A);
	kernel_1.setArg(1, buffer_B);
	kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype))); //local memory size

	//call all kernels in a sequence
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);

	//5.3 Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

	// Output Kernal execution time 
	std::cout << "Kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	
	// return result of calculation
	return ((float)B[0] / 10);
}

// Method used to calculate Maximum of values
float getMaximum(cl::Context context, cl::CommandQueue queue, cl::Program program) {
	
	typedef int mytype;
	cl::Event prof_event;

	//Part 4 - memory allocation
	//host - input
	//std::vector<mytype> A(10, 1);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
	vector<int> A = airTemp;

	//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
	//if the total input length is divisible by the workgroup size
	//this makes the code more efficient
	size_t local_size = 256;

	// adjust padding size according to the length of the array modulus the local size
	size_t padding_size = A.size() % local_size;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<int> A_ext(local_size - padding_size, 0);
		//append that extra vector to our input
		A.insert(A.end(), A_ext.begin(), A_ext.end());
	}

	size_t input_elements = A.size();//number of input elements
	size_t input_size = A.size() * sizeof(mytype);//size in bytes
	size_t nr_groups = input_elements / local_size;//define number of groups

	//host - output
	std::vector<mytype> B(input_elements);//initialse vector for output elements
	size_t output_size = B.size() * sizeof(mytype);//size in bytes

	//device - buffers
	// explained: http://github.khronos.org/OpenCL-CLHPP/classcl_1_1_buffer.html
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

	//Part 5 - device operations

	//5.1 copy array A to and initialise other arrays on device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

	//5.2 Setup and execute all kernels (i.e. device code)
	cl::Kernel kernel_1 = cl::Kernel(program, "maximum");
	kernel_1.setArg(0, buffer_A);
	kernel_1.setArg(1, buffer_B);
	kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype))); //local memory size

	//call all kernels in a sequence
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);

	//5.3 Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

	// output Kernal execution time
	std::cout << "Kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	
	// return result of calculation
	return ((float)B[0] / 10);
}

// Method used to calculate Sum and Average of values
float getAverage(cl::Context context, cl::CommandQueue queue, cl::Program program) {
	typedef int mytype;
	cl::Event prof_event;

	//Part 4 - memory allocation
	//host - input
	//std::vector<mytype> A(10, 1);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
	vector<int> A = airTemp;

	//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
	//if the total input length is divisible by the workgroup size
	//this makes the code more efficient
	size_t local_size = 32;

	// adjust padding size according to the length of the array modulus the local size
	size_t padding_size = A.size() % local_size;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<int> A_ext(local_size - padding_size, 0);
		//append that extra vector to our input
		A.insert(A.end(), A_ext.begin(), A_ext.end());
	}

	size_t input_elements = A.size();//number of input elements
	size_t input_size = A.size() * sizeof(mytype);//size in bytes
	size_t nr_groups = input_elements / local_size;//define number of groups

	//host - output
	std::vector<mytype> B(input_elements);//initialse vector for output elements
	size_t output_size = B.size() * sizeof(mytype);//size in bytes

	//device - buffers
	// explained: http://github.khronos.org/OpenCL-CLHPP/classcl_1_1_buffer.html
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

	//Part 5 - device operations

	//5.1 copy array A to and initialise other arrays on device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

	//5.2 Setup and execute all kernels (i.e. device code)
	cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_4");
	kernel_1.setArg(0, buffer_A);
	kernel_1.setArg(1, buffer_B);
	kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype))); //local memory size

	//call all kernels in a sequence
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);

	//5.3 Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

	// Output Kernal execution time
	std::cout << "Kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

	// Return result of calculation
	return ((float)B[0] / 10);

}

// Method used to calculate Standard Deviation of values
float getStandardDeviation(cl::Context context, cl::CommandQueue queue, cl::Program program, float mean) {
	
	// initialise event objects
	typedef int mytype;
	cl::Event prof_event;

	//Part 4 - memory allocation
	//host - input
	//std::vector<mytype> A(10, 1);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
	vector<int> A = airTemp;
	

	//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
	//if the total input length is divisible by the workgroup size
	//this makes the code more efficient
	size_t local_size = 256;

	// adjust padding size according to the length of the array modulus the local size
	size_t padding_size = A.size() % local_size;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<int> A_ext(local_size - padding_size, 0);
		//append that extra vector to our input
		A.insert(A.end(), A_ext.begin(), A_ext.end());
	}

	size_t input_elements = A.size();//number of input elements
	size_t input_size = A.size() * sizeof(mytype);//size in bytes
	size_t nr_groups = input_elements / local_size;//define number of groups

	//host - output
	std::vector<mytype> B(input_elements); //initialse vector for output elements
	size_t output_size = B.size() * sizeof(mytype);//size in bytes

	//device - buffers
	// explained: http://github.khronos.org/OpenCL-CLHPP/classcl_1_1_buffer.html
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

	//Part 5 - device operations

	//5.1 copy array A to and initialise other arrays on device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

	//5.2 Setup and execute all kernels (i.e. device code)
	cl::Kernel kernel_1 = cl::Kernel(program, "standardDeviation");
	kernel_1.setArg(0, buffer_A);
	kernel_1.setArg(1, buffer_B);
	kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype))); //local memory size
	kernel_1.setArg(3, mean); // mean passed as argument

	//call all kernels in a sequence
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);

	//5.3 Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

	// Output Kernal execution time
	std::cout << "Kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	
	// Return result of calculation by pulling first element within array
	return (float)B[0];
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	//
	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// Read data in from file
		readData();

		//Console outputs
		// Use printf functionality output items to decimal places
		// src: http://www.cplusplus.com/reference/cstdio/printf/
		std::cout << "\n*********************" << std::endl;
		float sum = getAverage(context, queue, program);
		printf("Total Sum = %.4f", sum);
		std::cout << "\n*********************" << std::endl;

		std::cout << "\n*********************" << std::endl;
		float average = sum / airTemp.size();
		printf("Mean (average) = %.7f", average);
		std::cout << "\n*********************" << std::endl;
		
		std::cout << "\n*********************" << std::endl;
		float minmum = getMinimum(context, queue, program);
		printf("Minimum = %.2f", minmum);
		std::cout << "\n*********************" << std::endl;

		std::cout << "\n*********************" << std::endl;
		float maximum = getMaximum(context, queue, program);
		printf("Maximum = %.2f", maximum);
		std::cout << "\n*********************" << std::endl;

		// Function returns partial Sum Squared Difference
		// Returns sum after initial operations, up to point of items in array whereby they are summed
		// Need to: 1) Divide result by original mean, 2) Square Root of the resultant
		std::cout << "\n*********************" << std::endl;
		float sdSum = getStandardDeviation(context, queue, program, average * 10);		
		std::cout << "Standard Deviation = " << (sqrt((sdSum / initialSize) / 10)) << std::endl;
		std::cout << "*********************" << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	return 0;
}
