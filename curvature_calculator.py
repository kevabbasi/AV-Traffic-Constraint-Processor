import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- STEP 1: Load the Raw Ego Motion Data ---
# REPLACE the file path below with the EXACT location of your downloaded Parquet file
FILE_PATH = "25cd4769-5dcf-4b53-a351-bf2c5deb6124.egomotion.parquet" 

try:
    df_ego = pd.read_parquet(FILE_PATH)
    print(f"Successfully loaded Ego Motion data with {len(df_ego)} time steps.")

except FileNotFoundError:
    print(f"ERROR: File not found at {FILE_PATH}. Please ensure you have downloaded and extracted the raw ego motion file for your target clip.")
    exit()

# --- STEP 2: Define and Apply the Curvature Feature Function ---
def calculate_curvature_feature(df_ego: pd.DataFrame) -> pd.DataFrame:
    """Calculates instantaneous roadway curvature (kappa) from ego motion data."""
    
    # 1. Prepare data for difference calculations
    df_ego = df_ego.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Convert quaternion (qx, qy, qz, qw) to yaw angle
    # Yaw is the rotation around the z-axis (vertical axis)
    # For quaternion, yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    df_ego['yaw'] = np.arctan2(
        2 * (df_ego['qw'] * df_ego['qz'] + df_ego['qx'] * df_ego['qy']),
        1 - 2 * (df_ego['qy']**2 + df_ego['qz']**2)
    )
    
    # 3. Calculate velocity magnitude from velocity components
    df_ego['velocity'] = np.sqrt(df_ego['vx']**2 + df_ego['vy']**2 + df_ego['vz']**2)
    
    # 4. Calculate change in Yaw (delta_yaw)
    # np.unwrap handles 360-degree angle transitions smoothly, necessary for vehicle control math.
    df_ego['yaw_unwrapped'] = np.unwrap(df_ego['yaw'].values)
    df_ego['delta_yaw'] = df_ego['yaw_unwrapped'].diff().fillna(0)
    
    # 5. Calculate time difference (delta_t in seconds)
    # Timestamp appears to be in microseconds, convert to seconds
    df_ego['delta_t'] = df_ego['timestamp'].diff().fillna(1e-5) / 1e6
    
    # 6. Calculate Yaw Rate (radians/second) and Curvature (kappa)
    df_ego['yaw_rate'] = df_ego['delta_yaw'] / df_ego['delta_t']
    
    # Curvature (kappa) = Yaw Rate / Velocity. (Core geometric constraint)
    epsilon = 0.01 
    df_ego['curvature_feature'] = df_ego['yaw_rate'] / (df_ego['velocity'].abs() + epsilon)
    
    return df_ego

# Apply the function to your data
df_ego = calculate_curvature_feature(df_ego)

# --- STEP 3: Print and Validate the Result ---
print("\n--- CURVATURE FEATURE CREATED ---")
print("These values represent the tightness of the road curve over time.")
print("\nCalculated curvature vs. existing curvature column:")
print(df_ego[['timestamp', 'velocity', 'yaw', 'curvature_feature', 'curvature']].head(10))
print("\nNote: The 'curvature' column already exists in the data.")
print("The 'curvature_feature' is our calculated version for comparison.")

# --- 1. Visualize Curvature Over Time ---

# Set the target file name for the final output (your deliverable)
FINAL_CSV_NAME = "Curvature_Feature_Analysis_Final.csv"

# Create the plot figure
plt.figure(figsize=(12, 5)) 

# Plot your custom feature (Curvature_Feature)
# We plot it against the time steps
plt.plot(df_ego['curvature_feature'], label='Calculated Curvature Feature (Your Code)', color='blue')

# Plot the ground truth data for comparison
plt.plot(df_ego['curvature'], label='Ground Truth Curvature', linestyle='--', alpha=0.6, color='red')

# Add labels and title
plt.title(f"Roadway Curvature Profile - Sample Analysis")
plt.xlabel("Time Step Index (approx. 10 Hz)")
plt.ylabel("Curvature (rad/m)")
plt.legend()
plt.grid(True)

# Save the plot image
plt.savefig("Curvature_Profile_Plot.png")
print("\nPlot saved successfully as Curvature_Profile_Plot.png")

# 2. Save Final Results to CSV (The Deliverable)
df_ego.to_csv(FINAL_CSV_NAME, index=False)
print(f"Final results saved to {FINAL_CSV_NAME}")

# Display the plot
plt.show()

# --- 1. Identify Start and End Time Steps ---

# The deep drop on the plot occurs between index 1000 and 1500.
START_INDEX = 1000
END_INDEX = 1500

# --- 2. Extract Raw Microsecond Timestamps ---
try:
    # Get the raw timestamp value (in microseconds) from the DataFrame rows
    start_timestamp_us = df_ego.loc[START_INDEX, 'timestamp']
    end_timestamp_us = df_ego.loc[END_INDEX, 'timestamp']

except KeyError:
    # This handles cases where the DataFrame might not use a perfect 0-based index
    print("ERROR: Could not find timestamps at the specified index. Ensure the DataFrame index starts at 0.")
    exit()

# --- 3. Calculate Time in Seconds ---
# The timestamps are relative to the start of the clip.
# We assume the clip starts near timestamp 0 (or use the minimum timestamp).

clip_start_us = df_ego['timestamp'].min()
event_start_time_s = (start_timestamp_us - clip_start_us) / 1_000_000
event_end_time_s = (end_timestamp_us - clip_start_us) / 1_000_000
event_duration_s = event_end_time_s - event_start_time_s

# --- 4. Print Instructions ---
print("\n--- VIDEO CUE POINTS ---")
print(f"Total event duration (time steps 1000 to 1500): {event_duration_s:.2f} seconds.")
print("-" * 30)
print(f"EVENT START TIME: {event_start_time_s:.2f} seconds")
print(f"EVENT END TIME: {event_end_time_s:.2f} seconds")
print("-" * 30)

# 5. Provide video file name instructions
# The video filename uses the same UUID you used for the Parquet file.
# Extract UUID from the file path
TARGET_UUID = os.path.splitext(os.path.basename(FILE_PATH))[0].replace('.egomotion', '')
print(f"Your video file name is: {TARGET_UUID}.camera_front_wide_120fov.mp4 (or similar camera view)")

