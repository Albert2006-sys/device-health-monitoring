// constants.h
#ifndef CONSTANTS_H
#define CONSTANTS_H

// 1. The Anomaly Threshold (from anomaly_threshold.npy)
const float ANOMALY_THRESHOLD = 0.448504; 

// 2. Normalization Constants (from scaler_mean.npy and scaler_scale.npy)
// There should be exactly 52 values in each array
const float SCALER_MEAN[52] = { /* Paste 52 values from scaler_mean.npy here */ };
const float SCALER_SCALE[52] = { /* Paste 52 values from scaler_scale.npy here */ };

// 3. Labels (from label_map.json)
const char* LABELS[] = {
    "Healthy_Idle", "Low_Oil", "Startup_Normal", "Combined_Fault_1",
    "Power_Steering", "Serpentine_Belt", "Brakes_Normal", "Worn_Brakes",
    "Bad_Ignition", "Dead_Battery", "Anomalous_Engine", "Testing_Export"
    // Ensure this matches the order in your label_map.json
};

#endif