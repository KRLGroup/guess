#! /bin/bash

black='\033[1;30m'
red='\033[1;31m'
green='\033[1;32m'
orange='\033[1;33m'
blue='\033[1;34m'
purple='\033[1;35m'
cyan='\033[1;36m'
lgray='\033[0;37m'
dgray='\033[1;30m'
white='\033[1;37m'
yellow='\033[1;33m'
nc='\033[0m'

bt=$(tput bold)
nt=$(tput sgr0)

print_usage() {
    echo -e "usage: ./guess_ros.sh [<map_name> default:simple_corridor] [<0/1> spin guesser default:1]"
    echo -e "available maps: simple_corridor - simple_corridor"
}

if [ "$1" == "-h" ] || [ "$1" == "-help" ]; then
    print_usage
    exit
fi

wait_sec() {
    local wait_msecs=$(($1*100))
    while [ $wait_msecs -gt 0 ]; do
	      if [ ${wait_msecs} -gt 99 ]; then
	          echo -ne "${bt}$2${nt} in ${bt}${yellow}$(echo ${wait_msecs} | cut -c1).$(echo ${wait_msecs} | cut -c2-3)${nc}${nt} secs\033[0K\r"
	      else
	          echo -ne "${bt}$2${nt} in ${bt}${yellow}0.$(echo ${wait_msecs} | cut -c1-2)${nc}${nt} secs\033[0K\r"
	      fi
	      sleep 0.01
	      : $((wait_msecs--))
    done

    echo -ne "\033[0K\r"
    echo -e "${bt}$2${nt}${nt}... ${green}done${nc}.${nt}"
}

# defaults
spin_guesser=1
map_name=simple_corridor

if [ $# -gt 0 ]; then
    map_name=$1
fi
if [ $# -gt 1 ]; then
    spin_guesser=$2
fi

GUESS_ROOT=/home/sapienzbot/ws/guess
pyscript=${GUESS_ROOT}/src/py/scan_guesser_socket.py
cmd_vel_topic=/cmd_vel

wait_sec 1 "Launching Guess-stage topological-navigation"
xterm -hold -e "rosrun scan_guesser_node topological_nav _map_name:=${map_name}" &

if [ ${spin_guesser} -eq 1 ]; then
    wait_sec 3 "Launching Guess-ros-node"
    xterm -hold -e "rosrun scan_guesser_node guesser _cmd_vel_topic:=${cmd_vel_topic}" &

    export PYTHONPATH="/usr/local/:${GUESS_ROOT}/src/py/:/usr/lib/python3/dist-packages/"
    python3 ${pyscript}
fi
