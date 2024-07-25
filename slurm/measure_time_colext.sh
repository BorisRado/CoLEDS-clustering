#!/bin/bash

# start the pods on all the devices
for device in jao1 jn1 jon1 jxn1; do
    sed "s/HOSTNAME/$device/g" pod_config.yaml | sed "s/BASEIMAGE/jetson/g" - | mk apply -f -
done

sed "s/HOSTNAME/lp1/g" pod_config.yaml | sed "s/BASEIMAGE/generic-cpu-x86/g" - | mk apply -f -

sed "s/HOSTNAME/op1/g" pod_config.yaml | sed "s/BASEIMAGE/generic-cpu-arm/g" - | mk apply -f -


all_pods_completed() {
  # Get the list of pod statuses
  pod_statuses=$(mk get pods --no-headers -o custom-columns=":status.phase")

  # Check if any pod is not in the 'Succeeded' or 'Failed' state
  for status in $pod_statuses; {
    if [[ "$status" != "Succeeded" && "$status" != "Failed" ]]; then
      echo "Pods are still running"
      echo ""
      return 1  # Return 1 (false) if any pod is still running or pending
    fi
  }

  return 0  # Return 0 (true) if all pods are completed
}


# Function to copy /home/data/temp from running pods to the local file system

copy_data_from_running_pods() {
  # Get the list of pod names and their statuses using jsonpath
  pod_info=$(mk get pods -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.phase}{"\n"}{end}')

  # Iterate through each pod name and copy the data if the pod is running
  echo "$pod_info" | while read -r pod_name phase; do
    echo "$pod_name is in $phase phase"
    if [[ "$phase" == "Running" ]]; then
      local_dir="/home/radovib/fl_clustering/data/raw/$pod_name"
      mkdir -p "$local_dir"
      mk cp "$pod_name:data/raw/tmp_times/" "$local_dir"  # Copy the folder from the pod to the local directory

    fi
  done
  echo "Done copying..."
  echo ""
}

# start a loop that every 20 minutes copies the folders to local file system
while ! all_pods_completed; do
    echo "Iterating..."
    copy_data_from_running_pods
    sleep 120
done
