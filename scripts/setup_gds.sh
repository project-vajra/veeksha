#!/bin/bash

# --- Configuration ---
DEFAULT_FS_TYPE="xfs"
DEFAULT_MOUNT_BASE_NAME="gds"
DEFAULT_MOUNT_PARENT_DIR="/mnt"
# --- End Configuration ---

# --- Helper Functions ---
print_info() {
    echo -e "\033[0;36m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARN]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
}

# Exit on errors
set -e

# --- Check if running as root ---
if [[ "$EUID" -ne 0 ]]; then
  print_error "This script must be run as root (or using sudo)."
  exit 1
fi

# --- Identify NVMe Drives ---
print_info "Detecting NVMe drives..."
# List block devices, show NAME/SIZE/TYPE, filter nvme, exclude partitions (like nvme0n1p1)
mapfile -t AVAILABLE_NVME < <(lsblk -dno NAME,SIZE,TYPE -p | awk '$3 == "disk" && $1 ~ /^\/dev\/nvme[0-9]+n[0-9]+$/ { print $1 " (" $2 ")" }')

if [ ${#AVAILABLE_NVME[@]} -eq 0 ]; then
    print_error "No NVMe drives detected (expected /dev/nvmeXnY format). Exiting."
    exit 1
fi

echo "Available NVMe drives:"
PS3="Select drive(s) (space-separated numbers, 'a' for all, 'q' to quit): "
select drive_choice in "${AVAILABLE_NVME[@]}" "All of the above" "Quit"; do
    case $drive_choice in
        "Quit")
            print_info "Quitting."
            exit 0
            ;;
        "All of the above")
            SELECTED_NVME_INFO=("${AVAILABLE_NVME[@]}")
            break
            ;;
        *)
            if [[ -n "$drive_choice" ]]; then
                # Handle multiple selections based on REPLY indices
                SELECTED_NVME_INFO=()
                for index in $REPLY; do
                   if [[ "$index" -ge 1 && "$index" -le ${#AVAILABLE_NVME[@]} ]]; then
                       SELECTED_NVME_INFO+=("${AVAILABLE_NVME[$index-1]}")
                   else
                       echo "Invalid selection: $index"
                   fi
                done
                if [[ ${#SELECTED_NVME_INFO[@]} -gt 0 ]]; then
                     break
                else
                    echo "No valid drives selected. Try again."
                fi
            else
                echo "Invalid choice. Try again."
            fi
            ;;
    esac
done

if [ ${#SELECTED_NVME_INFO[@]} -eq 0 ]; then
    print_error "No NVMe drives selected. Exiting."
    exit 1
fi

# Extract just the device paths
SELECTED_NVME_DEVS=()
echo "You selected:"
for info in "${SELECTED_NVME_INFO[@]}"; do
    dev_path=$(echo "$info" | awk '{print $1}')
    SELECTED_NVME_DEVS+=("$dev_path")
    echo "  - $info"
done

# --- Get Filesystem Type ---
read -p "Enter filesystem type [${DEFAULT_FS_TYPE}]: " FS_TYPE
FS_TYPE=${FS_TYPE:-$DEFAULT_FS_TYPE}
FS_TYPE=$(echo "$FS_TYPE" | tr '[:upper:]' '[:lower:]') # Convert to lowercase

if [[ "$FS_TYPE" != "xfs" && "$FS_TYPE" != "ext4" ]]; then
    print_error "Unsupported filesystem type '$FS_TYPE'. Please choose 'xfs' or 'ext4'."
    exit 1
fi
print_info "Using filesystem type: $FS_TYPE"
MKFS_CMD="mkfs.$FS_TYPE"
if ! command -v $MKFS_CMD &> /dev/null; then
    print_error "$MKFS_CMD command not found. Is the relevant package (e.g., xfsprogs, e2fsprogs) installed?"
    exit 1
fi

# --- Get Mount Point Base ---
read -p "Enter base name for mount points [${DEFAULT_MOUNT_BASE_NAME}]: " MOUNT_BASE_NAME
MOUNT_BASE_NAME=${MOUNT_BASE_NAME:-$DEFAULT_MOUNT_BASE_NAME}
MOUNT_PARENT_DIR=$(echo "$DEFAULT_MOUNT_PARENT_DIR" | sed 's:/*$::') # Remove trailing slash

# --- Confirmation ---
print_warning "*** WARNING: DATA LOSS IMMINENT! ***"
print_warning "The following NVMe drives will be COMPLETELY WIPED and formatted with '$FS_TYPE':"
for dev in "${SELECTED_NVME_DEVS[@]}"; do
    print_warning "  - $dev"
done
print_warning "Mount points will be created under '${MOUNT_PARENT_DIR}/${MOUNT_BASE_NAME}_<index>'."
print_warning "/etc/fstab will be modified."
read -p "Are you absolutely sure you want to proceed? (yes/no): " CONFIRMATION
if [[ "$CONFIRMATION" != "yes" ]]; then
    print_info "Operation cancelled by user."
    exit 0
fi

# --- Format, Create Mount Points, Mount ---
print_info "Starting filesystem preparation..."
MOUNT_POINTS=()
DRIVE_MOUNT_MAP=() # Store device -> mount_point mapping for fstab

for i in "${!SELECTED_NVME_DEVS[@]}"; do
    dev="${SELECTED_NVME_DEVS[$i]}"
    mount_point="${MOUNT_PARENT_DIR}/${MOUNT_BASE_NAME}_${i}"
    MOUNT_POINTS+=("$mount_point")
    DRIVE_MOUNT_MAP+=("${dev}:${mount_point}") # Store mapping

    print_info "Processing $dev -> $mount_point"

    # Unmount just in case it was previously mounted (ignore errors)
    umount "$dev" &> /dev/null || true

    print_info "  Formatting $dev with $FS_TYPE..."
    if [[ "$FS_TYPE" == "xfs" ]]; then
        # XFS often requires -f if it detects a previous FS
        $MKFS_CMD -f "$dev"
    else
        $MKFS_CMD "$dev"
    fi

    print_info "  Creating mount point $mount_point..."
    mkdir -p "$mount_point"

    print_info "  Mounting $dev on $mount_point..."
    mount "$dev" "$mount_point"
done

# --- Update /etc/fstab ---
FSTAB="/etc/fstab"
FSTAB_BACKUP="${FSTAB}.backup.$(date +%F_%T)"

print_info "Backing up $FSTAB to $FSTAB_BACKUP..."
cp "$FSTAB" "$FSTAB_BACKUP"

print_info "Adding entries to $FSTAB..."
for mapping in "${DRIVE_MOUNT_MAP[@]}"; do
    dev="${mapping%%:*}"       # Get part before :
    mount_point="${mapping#*:}" # Get part after :

    # Check if entry for device already exists (basic check)
    if grep -q "^\s*${dev}\s" "$FSTAB"; then
        print_warning "  Entry for ${dev} might already exist in $FSTAB. Skipping add. Please verify manually."
    else
        print_info "  Adding: ${dev} ${mount_point} ${FS_TYPE} defaults,nofail 0 0"
        # Append the new entry
        echo "${dev}   ${mount_point}   ${FS_TYPE}   defaults,nofail   0   0" >> "$FSTAB"
    fi
done

# --- Verify Mounts ---
print_info "Attempting to mount all filesystems listed in $FSTAB..."
if mount -a; then
    print_info "Mount -a succeeded."
else
    print_warning "Mount -a reported errors. Please check $FSTAB and system logs ('journalctl' or '/var/log/syslog')."
    print_warning "Your original fstab is backed up at: $FSTAB_BACKUP"
fi

print_info "---------------------------------------------"
print_info "Automation of Step 4 completed."
print_info "Configured devices:"
df -hT "${MOUNT_POINTS[@]}"
print_info "Please review $FSTAB and the backup $FSTAB_BACKUP."
print_info "---------------------------------------------"

exit 0
