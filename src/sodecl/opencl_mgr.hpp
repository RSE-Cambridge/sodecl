#ifndef SODECL_OPENLC_MGR_HPP
#define SODECL_OPENLC_MGR_HPP

#include "../sodecl.hpp"
#include <CL/cl.hpp>
#include <stdexcept>

namespace sodecl {

/**
 * @brief Class that implements the OpenCL functions needed by the ODE and SDE solvers.
 * 
 */
    class opencl_mgr {
    private:
        /********************************************************************************************
        * OPENCL HARDWARE SECTION VARIABLES
        */

        cl_uint opencl_platform_count;    /**< Number of OpenCL platforms */
        std::vector<sodecl::platform *> opencl_platforms; /**< Vector which stores the sodecl::platform objects. One object for each OpenCL platform */
        cl_uint selected_platform; /**< The index of the selected sodecl::platform object in the m_platforms vector */
        cl_uint selected_device;   /**< The index of the selected sodecl::device object in m_devices vector of selected platform */
        device_Type selected_device_type;  /**< Selected OpenCL device type */

        /********************************************************************************************
        OPENCL SOFTWARE SECTION VARIABLES
        */

        std::vector <cl_context> contexts;    /**< OpenCL command contexts vector */
        std::vector <cl_command_queue> command_queues;    /**< OpenCL command queues vector */
        std::vector<char> kernel_sources; /**< Char vector which stores the OpenCL kernel source string. @todo store multiple kernel source strings */
        std::string build_options_str;    /**< Char vector which stores the OpenCL build options string */
        std::vector <build_Option> build_options; /**< build_Option vector which stores the OpenCL build options selection */
        char *source_str; /**< OpenCL kernel string */
        string kernel_path_str;   /**< OpenCL kernels solvers path */
        size_t source_size;   /**< OpenCL kernel string size */
        std::vector <cl_program> programs;    /**< OpenCL programs vector */
        std::vector <cl_kernel> kernels;  /**< OpenCL kernels vector */
        int local_group_size; /**< OpenCL device local group size */

        // Log mechanisms
        clog *m_log; /**< Pointer for log */

    public:
        /**
         * @brief Default constructor.
         *
         */
        opencl_mgr() {
            // Initialise the clog object
            m_log = clog::getInstance();
        };

        /**
         * @brief Destructor.
         *
         */
        ~opencl_mgr() {
            for (auto i : opencl_platforms) {
                delete i;
            }
            opencl_platforms.clear();
        };

        /**
         * @brief Returns the number of OpenCL platforms available.
         *
         * @return      int         The number of OpenCL platforms available. Returns -1 if the operation failed.
         */
        int get_opencl_platform_count() {
            // get platform count
            cl_int err = clGetPlatformIDs(0, NULL, &opencl_platform_count);

            if (err == CL_INVALID_VALUE) {
                throw std::invalid_argument("Supplied values to the function for getting the OpenCL platform IDs are invalid.\n");
            }

            if (err == CL_OUT_OF_HOST_MEMORY) {
                throw std::runtime_error("There was a failure to allocate resources required by the OpenCL implementation on the host.\n");
            }

            return (int) opencl_platform_count;
        }

        /**
         * @brief Creates all sodecl::platform objects.
         *
         * @return      int         The number of OpenCL platforms available. Returns -1 if the operation failed.
         */
        int create_opencl_platforms() {
            cl_platform_id *cpPlatform = new cl_platform_id[opencl_platform_count];

            // get all OpenCL platforms
            cl_int err = clGetPlatformIDs(opencl_platform_count, cpPlatform, NULL);

            if (err == CL_INVALID_VALUE) {
                throw std::runtime_error("Supplied values to the function for getting the OpenCL platform IDs are invalid.\n");
            }

            if (err == CL_OUT_OF_HOST_MEMORY) {
                throw std::runtime_error("There was a failure to allocate resources required by the OpenCL implementation on the host.\n");
            }

            for (cl_uint i = 0; i < opencl_platform_count; i++) {
                opencl_platforms.push_back(new platform(cpPlatform[i]));
            }
            delete[] cpPlatform;

            return 1;
        }

        /**
         * @brief Sets the sodecl object to use the selected OpenCL device for the integration of the ODE model.
         *
         * @param	platform_num	Index of selected OpenCL platform
         * @param	device_type		OpenCL device type
         * @param	device_num		Index of selected OpenCL device in the selected OpenCL platform
         * @return  int             Returns 1 if the operations were succcessful or 0 if they were unsuccessful.
         */
        int choose_opencl_device(cl_uint platform_num, device_Type device_type, cl_uint device_num) {
            // Check if selected platform exist
            if ((int) platform_num < 0 || (int) platform_num > opencl_platform_count) {
                //cerr << "Selected platform number is out of bounds." << std::endl;
                m_log->write("Selected platform number is out of bounds.\n");
                return 0;
            }

            // Check if selected device exist
            if ((int) device_num < 0 || (int) device_num > opencl_platforms[platform_num]->get_device_count()) {
                //cerr << "Selected device number is out of bounds." << std::endl;
                m_log->write("Selected device number is out of bounds.\n");
                return 0;
            }

            // If selected device type is not ALL (OpenCL type) check if selected device type exist
            if ((cl_device_type) device_type != (cl_device_type) device_Type::ALL) {
                if (opencl_platforms[platform_num]->m_devices[device_num]->type() != (cl_device_type) device_type) {
                    //cerr << "Selected device is not of the type selected." << std::endl;
                    m_log->write("Selected device is not of the type selected.\n");
                    return 0;
                }
            }

            //std::cout << "Selected platform: " << m_platforms[platform_num]->name().c_str() << std::endl;
            //std::cout << "Device name: " << m_platforms[platform_num]->m_devices[device_num]->name().c_str() << std::endl;
            //std::cout << "Device type: " << m_platforms[platform_num]->m_devices[device_num]->type() << std::endl;

            m_log->write("Selected platform name: ");
            m_log->write(opencl_platforms[platform_num]->name().c_str());
            m_log->write("\n");

            m_log->write("Selected device name: ");
            m_log->write(opencl_platforms[platform_num]->m_devices[device_num]->name().c_str());
            m_log->write("\n");

            m_log->write("Selected device OpenCL version: ");
            m_log->write(opencl_platforms[platform_num]->m_devices[device_num]->version().c_str());
            m_log->write("\n");

            selected_platform = platform_num;
            selected_device = device_num;
            selected_device_type = device_type;

            return 1;
        }

        /**
         * @brief Create OpenCL context for the selected OpenCL platform and OpenCL device.
         *
         * @return  int  Returns 1 if the operations were succcessful or 0 if they were unsuccessful.
         */
        int create_opencl_context() {
            cl_int err;
            cl_context context = clCreateContext(NULL, 1,
                                                 &(opencl_platforms[selected_platform]->m_devices[selected_device]->m_device_id),
                                                 NULL, NULL, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Error: Failed to create context! " << err << std::endl;
                return 0;
            }
            contexts.push_back(context);

            return 1;
        }
    };
}

#endif // SODECL_OPENLC_MGR_HPP