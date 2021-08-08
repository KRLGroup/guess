#! /bin/bash

print_usage() {
    echo -e "usage: ./guess_env.sh [<map_name> default:simple_corridor]"
    echo -e "available maps: simple_corridor - diag_floor_b1"
}

if [ "$1" == "-h" ] || [ "$1" == "-help" ]; then
    print_usage
    exit
fi

if [ $# -lt 1 ]; then
    map_name=diag_floor_b1
else
    map_name=$1
fi

GUESS_ROOT=/home/shrijitsingh99/guess
rcfg=${GUESS_ROOT}/config/rviz/stage_guess.config.rviz
cmd_vel_topic=/cmd_vel
eval "source ~/guess/devel/setup.bash"
echo -e "Launching Guess-stage environment"
eval "roslaunch scan_guesser_node guess_stage_env.launch worldfile:=${map_file} map_name:=${map_name} cmd_vel_topic:=${cmd_vel_topic} rviz_cfg:=${rcfg}" 
