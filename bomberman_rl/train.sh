#!/bin/bash

: '
Usage of this script:
- first make sure you activate the script, Do this with: chmod -x train.sh
- Afterwards you can type ./train.sh 5 150
    -> This will create 5 training sessions, i.e. 5 history files
    -> Each agent will play 150 rounds in every session
'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

n=$1        #n gives the number of training sessions -> number of histories
rounds=$2   #rounds says how many rounds the agent should train
for run in $(eval echo {1..$n})
do
    printf "${BLUE}****************************\n ${GREEN}started training number $run ${NC}\n"
    python main.py play --agents linear_agent_coin --train 1 --n-rounds $rounds --no-gui
    printf "\n${GREEN} finished training number $run \n"
done
printf "${GREEN}\n ... done! \n${BLUE}****************************${NC}"

