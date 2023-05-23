#!/bin/bash 


if [ -z $1 ]; then
  echo "No username argument given"
  exit 1 
fi 


USERNAME=$1
echo "Logging in with username $USERNAME"
ssh $USERNAME@puhti.csc.fi
