#!/bin/bash 


if [ -z $1 ]; then
  echo "No username argument given"
  USERNAME="ollijoki"
  echo "Using $USERNAME as username"
else
 USERNAME=$1
fi

cat ~/UniHY/keys/puhti_password.txt



echo "Logging in with username $USERNAME"
ssh $USERNAME@puhti.csc.fi
