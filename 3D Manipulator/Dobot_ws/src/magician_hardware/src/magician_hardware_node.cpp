#include <magician_hardware/magician_hardware_interface.h>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

typedef struct{
    controller_manager::ControllerManager *manager;
    magician_hardware::MagicianHWInterface *interface;
}ArgsForThread;

static void timespecInc(struct timespec &tick, int nsec)
{
  int SEC_2_NSEC = 1e+9;
  tick.tv_nsec += nsec;
  while (tick.tv_nsec >= SEC_2_NSEC)
  {
    tick.tv_nsec -= SEC_2_NSEC;
    ++tick.tv_sec;
  }
}

static boost::mutex reset_pose_mutex;
static boost::mutex stop_update_mutex;
static bool reseting_pose;
static bool stopping_update;

static controller_manager::ControllerManager* ctlr_maganer_ptr;

bool magician_hardware::MagicianHWInterface::ResetPose(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &resp)
{
    std::vector<std::string> start_list;
    std::vector<std::string> stop_list;

    start_list.clear();
    stop_list.clear();
    stop_list.push_back("magician_arm_controller");

    ctlr_maganer_ptr->switchController(start_list, stop_list, 2);

    boost::mutex::scoped_lock reset_pose_lock(reset_pose_mutex);
    reseting_pose=true;
    reset_pose_lock.unlock();

    ros::Rate r(10);
    r.sleep();

    bool stop_update=false;
    boost::mutex::scoped_lock stop_update_lock(stop_update_mutex);
    stop_update=stopping_update;
    stop_update_lock.unlock();

    if(!stop_update)
    {
        resp.message="Failed, robot may be moving";
        resp.success=false;
        boost::mutex::scoped_lock reset_pose_lock(reset_pose_mutex);
        reseting_pose=false;
        reset_pose_lock.unlock();

        stop_list.clear();
        start_list.push_back("magician_arm_controller");

        ctlr_maganer_ptr->switchController(start_list, stop_list, 2);

        return true;
    }

    std::vector<double> jnt_values;
    bool result=magician_device_->ResetPose(req, resp, jnt_values);

    if(!resp.success)
    {
        boost::mutex::scoped_lock reset_pose_lock(reset_pose_mutex);
        reseting_pose=false;
        reset_pose_lock.unlock();

        stop_list.clear();
        start_list.push_back("magician_arm_controller");

        ctlr_maganer_ptr->switchController(start_list, stop_list, 2);

        return true;
    }

    if(!reinitPose(jnt_values))
    {
        resp.message="Failed, the length of joint_values is wrong";
        resp.success=false;
        boost::mutex::scoped_lock reset_pose_lock(reset_pose_mutex);
        reseting_pose=false;
        reset_pose_lock.unlock();

        stop_list.clear();
        start_list.push_back("magician_arm_controller");

        ctlr_maganer_ptr->switchController(start_list, stop_list, 2);

        return true;
    }
    else
    {
        boost::mutex::scoped_lock reset_pose_lock(reset_pose_mutex);
        reseting_pose=false;
        reset_pose_lock.unlock();

        stop_list.clear();
        start_list.push_back("magician_arm_controller");

        ctlr_maganer_ptr->switchController(start_list, stop_list, 2);

        return true;
    }
}

void *update_loop(void *threadarg)
{
    ArgsForThread *arg=(ArgsForThread *)threadarg;
    controller_manager::ControllerManager *manager=arg->manager;
    magician_hardware::MagicianHWInterface *interface=arg->interface;
    ros::Duration d(0.016);
    struct timespec tick;
    clock_gettime(CLOCK_REALTIME, &tick);
    //time for checking overrun
    struct timespec before;
    double overrun_time;

    bool reset_pose=false;
    boost::mutex::scoped_lock reset_pose_lock(reset_pose_mutex);
    reset_pose_lock.unlock();
    boost::mutex::scoped_lock stop_update_lock(stop_update_mutex);
    stop_update_lock.unlock();

    while(ros::ok())
    {
        ros::Time this_moment(tick.tv_sec, tick.tv_nsec);

        reset_pose_lock.lock();
        reset_pose=reseting_pose;
        reset_pose_lock.unlock();

        if(reset_pose)
        {
            if(interface->isMoving())
            {
                reset_pose=false;
            }
        }

        if(!reset_pose)
        {
            interface->read(this_moment, d);
            manager->update(this_moment, d);
            interface->write(this_moment, d);
        }
        else
        {
            stop_update_lock.lock();
            stopping_update=true;
            stop_update_lock.unlock();
        }

        timespecInc(tick, d.nsec);
        // check overrun
        clock_gettime(CLOCK_REALTIME, &before);
        overrun_time = (before.tv_sec + double(before.tv_nsec)/1e+9) -  (tick.tv_sec + double(tick.tv_nsec)/1e+9);
        if(overrun_time > 0.0)
        {
            tick.tv_sec=before.tv_sec;
            tick.tv_nsec=before.tv_nsec;
            std::cout<<"overrun_time: "<<overrun_time<<std::endl;
        }

        clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &tick, nullptr);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        ROS_ERROR("[USAGE]Application portName");
        return -1;
    }
    // Connect Dobot before start the service
    int result = ConnectDobot(argv[1], 115200, nullptr, nullptr);
    switch (result) {
        case DobotConnect_NoError:
        break;
        case DobotConnect_NotFound:
            ROS_ERROR("Dobot not found!");
            return -2;
        break;
        case DobotConnect_Occupied:
            ROS_ERROR("Invalid port name or Dobot is occupied by other application!");
            return -3;
        break;
        default:
        break;
    }
    ros::init(argc, argv, "magician_hardware_node", ros::init_options::AnonymousName);

    magician_hardware::MagicianHWInterface magician_hw_interface;
    controller_manager::ControllerManager ctlr_manager(&magician_hw_interface);

    ctlr_maganer_ptr=&ctlr_manager;

    ros::NodeHandle n1, n2("~");
    magician_hw_interface.init(n1, n2);

    reseting_pose=false;
    stopping_update=false;

    ros::ServiceServer reset_pose_server=n2.advertiseService("reset_pose", &magician_hardware::MagicianHWInterface::ResetPose, &magician_hw_interface);

    pthread_t tid;
    ArgsForThread *thread_arg=new ArgsForThread();
    thread_arg->manager=&ctlr_manager;
    thread_arg->interface=&magician_hw_interface;

    pthread_create(&tid, nullptr, update_loop, thread_arg);

    ros::Rate r(100);
    while (ros::ok()) {
        ros::spinOnce();
        r.sleep();
    }
}
