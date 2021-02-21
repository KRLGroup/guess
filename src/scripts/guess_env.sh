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
    map_name=simple_corridor
else
    map_name=$1
fi

GUESS_ROOT=/home/sapienzbot/ws/guess
rcfg=${GUESS_ROOT}/config/rviz/stage_guess.config.rviz
map_file=${GUESS_ROOT}/config/world/${map_name}
# urdf_file=${GUESS_ROOT}/config/ros/urdf/
cmd_vel_topic=/cmd_vel

echo -e "Launching Guess-stage environment"
xterm -hold -e "roslaunch scan_guesser_node guess_stage_env.launch worldfile:=${map_file}.world mapfile:=${map_file}.yaml urdffile:=${urdffile} cmd_vel_topic:=${cmd_vel_topic} rviz_cfg:=${rcfg}" &
