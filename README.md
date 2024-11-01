# ESP32-CAM Face Recognition Door Lock System

This project implements a face recognition system using the ESP32-CAM module to control a door lock via a relay. The system captures images, detects and recognizes faces, and unlocks the door when an authorized face is recognized.

## Table of Contents

1. [Overview](#overview)
2. [Hardware Components](#hardware-components)
3. [Libraries and Definitions](#libraries-and-definitions)
4. [Global Variables and Constants](#global-variables-and-constants)
5. [Function Declarations](#function-declarations)
6. [Face Recognition Configuration](#face-recognition-configuration)
7. [Setup Function](#setup-function)
8. [Face Recognition Initialization](#face-recognition-initialization)
9. [Door Control Functions](#door-control-functions)
10. [Main Loop](#main-loop)
11. [Detailed Explanation of Face Detection and Recognition](#detailed-explanation-of-face-detection-and-recognition)
12. [Neural Network Usage](#neural-network-usage)
13. [Libraries and Functions](#libraries-and-functions)
14. [Conclusion](#conclusion)

## Overview

The system uses an ESP32-CAM module to capture images and perform face detection and recognition using neural networks. Authorized faces are stored in the flash memory, and when a recognized face is detected, the system unlocks the door by activating a relay.

## Hardware Components

- **ESP32-CAM Module:** Captures images and processes face recognition.
- **Relay Module:** Controls the door lock mechanism.
- **Power Supply:** Provides power to the ESP32-CAM and relay.

## Libraries and Definitions

```cpp
#include "esp_camera.h"        // Camera functions
#include "esp_timer.h"         // Timer functions
#include "Arduino.h"           // Core Arduino functions
#include "fd_forward.h"        // Face detection functions
#include "fr_forward.h"        // Face recognition functions
#include "fr_flash.h"          // Flash memory functions

#define ENROLL_CONFIRM_TIMES 5 // Number of confirmations for enrollment
#define FACE_ID_SAVE_NUMBER 100 // Maximum number of faces to store

#define CAMERA_MODEL_AI_THINKER // Camera model
#include "camera_pins.h"        // Camera pin definitions
```

- **esp_camera.h:** Provides functions for camera initialization and image capture.
- **esp_timer.h:** Used for timing operations.
- **fd_forward.h and fr_forward.h:** Provide face detection and recognition functionalities.
- **fr_flash.h:** Handles saving and loading face data from flash memory.
- **camera_pins.h:** Contains pin definitions specific to the AI Thinker ESP32-CAM module.

## Global Variables and Constants

```cpp
camera_fb_t * fb = NULL;        // Frame buffer

#define relay_pin 2             // GPIO pin connected to the relay
unsigned long door_opened_millis = 0; // Timestamp when the door was opened
long interval = 5000;           // Duration to keep the door unlocked (in milliseconds)
```

- **fb:** Pointer to the camera frame buffer.
- **relay_pin:** GPIO pin controlling the relay (door lock).
- **door_opened_millis:** Stores the time when the door was unlocked.
- **interval:** Time in milliseconds to keep the door unlocked after recognizing a face.

## Function Declarations

```cpp
void app_facenet_main();
```

- **app_facenet_main():** Initializes face recognition components.

## Face Recognition Configuration

### Image Processing Result Structure

```cpp
typedef struct
{
  uint8_t *image;               // Image data
  box_array_t *net_boxes;       // Detected face bounding boxes
  dl_matrix3d_t *face_id;       // Face ID features
} http_img_process_result;
```

- **image:** Pointer to the processed image data.
- **net_boxes:** Contains information about detected faces.
- **face_id:** Unique identifier extracted from a face image.

### MTCNN Configuration

```cpp
static inline mtmn_config_t app_mtmn_config()
{
  mtmn_config_t mtmn_config = {0}; // Initialize the configuration structure
  mtmn_config.type = FAST;         // Set detection type to FAST
  mtmn_config.min_face = 80;       // Minimum face size to detect (pixels)
  mtmn_config.pyramid = 0.707;     // Image scaling factor
  mtmn_config.pyramid_times = 4;   // Number of scaling steps

  // Thresholds for the P-Net (Proposal Network)
  mtmn_config.p_threshold.score = 0.6;
  mtmn_config.p_threshold.nms = 0.7;
  mtmn_config.p_threshold.candidate_number = 20;

  // Thresholds for the R-Net (Refinement Network)
  mtmn_config.r_threshold.score = 0.7;
  mtmn_config.r_threshold.nms = 0.7;
  mtmn_config.r_threshold.candidate_number = 10;

  // Thresholds for the O-Net (Output Network)
  mtmn_config.o_threshold.score = 0.7;
  mtmn_config.o_threshold.nms = 0.7;
  mtmn_config.o_threshold.candidate_number = 1;

  return mtmn_config;
}
mtmn_config_t mtmn_config = app_mtmn_config();    // Initialize MTCNN configuration
```

- **mtmn_config_t:** Configuration structure for the Multi-Task Cascaded Convolutional Neural Network (MTCNN) used in face detection.
- **type:** Sets the detection type to `FAST` for quicker detection.
- **Thresholds:** Define the sensitivity and accuracy of face detection at different stages (P-Net, R-Net, O-Net).

### Face ID List and Aligned Face Matrix

```cpp
face_id_name_list st_face_list;                   // Face ID list
static dl_matrix3du_t *aligned_face = NULL;       // Aligned face image matrix
```

- **st_face_list:** Stores the face IDs and associated names.
- **aligned_face:** Pointer to the aligned face image data.

## Setup Function

```cpp
void setup() {
  Serial.begin(115200);        // Start serial communication for debugging
  Serial.println();

  digitalWrite(relay_pin, LOW); // Ensure the relay is off (door locked)
  pinMode(relay_pin, OUTPUT);   // Set relay pin as output
```

- Initializes serial communication and sets up the relay control pin.

### Camera Configuration

```cpp
  camera_config_t config;       // Camera configuration structure
  config.ledc_channel = LEDC_CHANNEL_0; // LEDC PWM channel for camera clock
  config.ledc_timer = LEDC_TIMER_0;     // LEDC timer for camera clock

  // Camera pin assignments
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;       // Set XCLK frequency to 20MHz
  config.pixel_format = PIXFORMAT_JPEG; // Set pixel format to JPEG
```

- **camera_config_t config:** Holds the configuration settings for the camera.
- **Pin Assignments:** Map the ESP32-CAM pins to the camera module's pins.
- **xclk_freq_hz:** Sets the external clock frequency for the camera.
- **pixel_format:** Sets the format of the captured images.

### Frame Size and Buffer Configuration

```cpp
  if (psramFound()) {                   // Check if PSRAM is available
    config.frame_size = FRAMESIZE_UXGA; // Set high resolution if PSRAM is available
    config.jpeg_quality = 10;           // JPEG quality (lower is better)
    config.fb_count = 2;                // Use two frame buffers
  } else {
    config.frame_size = FRAMESIZE_SVGA; // Set lower resolution if no PSRAM
    config.jpeg_quality = 12;           // JPEG quality
    config.fb_count = 1;                // Use one frame buffer
  }
```

- **psramFound():** Detects if external PSRAM is available on the ESP32.
- **frame_size:** Determines the resolution of captured images.
- **fb_count:** Number of frame buffers to use.

### Camera Initialization

```cpp
  esp_err_t err = esp_camera_init(&config); // Initialize the camera
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get(); // Get camera sensor handle
  s->set_framesize(s, FRAMESIZE_QVGA);    // Set frame size to QVGA (320x240)

  app_facenet_main();                     // Initialize face recognition components
}
```

- **esp_camera_init():** Initializes the camera with the specified configuration.
- **esp_camera_sensor_get():** Retrieves the sensor handle to modify settings.
- **set_framesize():** Sets the frame size to QVGA for face detection.

## Face Recognition Initialization

```cpp
void app_facenet_main()
{
  face_id_name_init(&st_face_list, FACE_ID_SAVE_NUMBER, ENROLL_CONFIRM_TIMES); // Initialize face ID list
  aligned_face = dl_matrix3du_alloc(1, FACE_WIDTH, FACE_HEIGHT, 3);            // Allocate memory for aligned face
  read_face_id_from_flash_with_name(&st_face_list);                            // Load stored face IDs from flash
}
```

- **face_id_name_init():** Initializes the face ID list with the specified capacity and enrollment confirmations.
- **dl_matrix3du_alloc():** Allocates memory for storing the aligned face image.
- **read_face_id_from_flash_with_name():** Loads saved face IDs and names from flash memory into `st_face_list`.

## Door Control Functions

### Open Door Function

```cpp
void open_door() {
  if (digitalRead(relay_pin) == LOW) {
    digitalWrite(relay_pin, HIGH); // Activate the relay to unlock the door
    Serial.println("Door Unlocked");
    door_opened_millis = millis(); // Record the time when the door was unlocked
  }
}
```

- **open_door():** Activates the relay to unlock the door and records the unlock time.

## Main Loop

```cpp
void loop() {
  dl_matrix3du_t *image_matrix = dl_matrix3du_alloc(1, 320, 240, 3); // Allocate image matrix
  http_img_process_result out_res = {0};     // Initialize image processing result
  out_res.image = image_matrix->item;        // Assign image data pointer

  fb = esp_camera_fb_get();                  // Capture a frame

  if (fb)
  {
    fmt2rgb888(fb->buf, fb->len, fb->format, out_res.image); // Convert image to RGB888

    out_res.net_boxes = face_detect(image_matrix, &mtmn_config); // Detect faces

    if (out_res.net_boxes)
    {
      if (align_face(out_res.net_boxes, image_matrix, aligned_face) == ESP_OK) // Align face
      {
        out_res.face_id = get_face_id(aligned_face); // Extract face ID

        if (st_face_list.count > 0)
        {
          face_id_node *f = recognize_face_with_name(&st_face_list, out_res.face_id); // Recognize face
          if (f)
          {
            Serial.printf("Face recognized: %s\n", f->id_name); // Print recognized face name
            open_door();                                        // Unlock door
          }
          else
          {
            Serial.println("Face not recognized"); // Face not recognized
          }
        }
        dl_matrix3d_free(out_res.face_id); // Free face ID memory
      }
    }
    esp_camera_fb_return(fb); // Return the frame buffer
    fb = NULL;                // Reset frame buffer pointer
  }

  // Lock the door after the interval has passed
  if (millis() - interval > door_opened_millis) { 
    digitalWrite(relay_pin, LOW); // Deactivate the relay to lock the door
  }
  
  dl_matrix3du_free(image_matrix); // Free image matrix memory
}
```

- **Image Capture and Processing:**
  - Captures an image frame.
  - Converts the image to RGB888 format.
  - Detects faces in the image.
- **Face Alignment and Recognition:**
  - Aligns the detected face.
  - Extracts a unique face ID using a neural network.
  - Compares the extracted face ID with stored face IDs.
- **Door Control:**
  - Unlocks the door if a recognized face is detected.
  - Locks the door after a specified interval.
- **Memory Management:**
  - Frees allocated memory to prevent leaks.

## Detailed Explanation of Face Detection and Recognition

### Face Detection with MTCNN

- **Multi-Task Cascaded Convolutional Neural Network (MTCNN):** A neural network framework used for face detection.
- **Stages of MTCNN:**
  - **P-Net (Proposal Network):** Generates candidate face regions quickly.
  - **R-Net (Refinement Network):** Refines the candidate regions.
  - **O-Net (Output Network):** Produces final face bounding boxes and landmarks.
- **Configuration Parameters:**
  - **Score Thresholds:** Determine the confidence level required to consider a detection valid.
  - **Non-Maximum Suppression (NMS):** Eliminates overlapping bounding boxes to reduce false positives.
  - **Candidate Numbers:** Limits the number of proposals to improve performance.

### Face Alignment

- **Purpose:** Standardizes the position, size, and orientation of the face for better recognition accuracy.
- **Process:** Uses facial landmarks (e.g., eyes, nose) to align the face image.

### Face Recognition with Neural Networks

- **Feature Extraction:**
  - The aligned face image is processed by a neural network to extract a face ID (feature vector).
  - **get_face_id():** Function that returns a numerical representation of the face.
- **Face Matching:**
  - **recognize_face_with_name():** Compares the extracted face ID with stored IDs.
  - **Similarity Metrics:** Uses Euclidean distance or cosine similarity to determine if the faces match.
- **Enrollment Confirmation:**
  - Multiple confirmations are required during enrollment to ensure the accuracy of the stored face ID.

## Neural Network Usage

- **Pre-trained Models:** The system uses pre-trained neural networks for both face detection and recognition.
- **Face Detection Network:** MTCNN is used to detect faces in images.
- **Face Recognition Network:** Extracts unique features from faces to create a face ID.
- **Feature Vectors:** High-dimensional numerical representations that uniquely identify a face.

## Libraries and Functions

### Key Libraries

- **esp_camera.h:** Camera initialization and image capture functions.
- **esp_timer.h:** Timing functions, such as `millis()`.
- **fd_forward.h and fr_forward.h:** Face detection and recognition functionalities.
- **fr_flash.h:** Functions for reading and writing face data to flash memory.

### Important Functions

- **Camera Functions:**
  - **esp_camera_init():** Initializes the camera.
  - **esp_camera_fb_get():** Captures an image frame.
  - **esp_camera_fb_return():** Returns the frame buffer.

- **Image Processing Functions:**
  - **fmt2rgb888():** Converts image data to RGB888 format.
  - **dl_matrix3du_alloc():** Allocates memory for image data.
  - **dl_matrix3du_free():** Frees allocated image memory.

- **Face Detection and Recognition Functions:**
  - **face_detect():** Detects faces in an image.
  - **align_face():** Aligns a face image.
  - **get_face_id():** Extracts a face ID from an aligned face.
  - **recognize_face_with_name():** Compares a face ID with stored IDs.

- **Flash Memory Functions:**
  - **read_face_id_from_flash_with_name():** Loads face IDs from flash memory.
  - **face_id_name_init():** Initializes the face ID list.

### Memory Management

- **dl_matrix3du_alloc() and dl_matrix3du_free():** Manage memory allocation and deallocation for images.
- **dl_matrix3d_free():** Frees memory allocated for face ID data.

## Conclusion

This code provides a comprehensive solution for a face recognition door lock system using the ESP32-CAM module. By leveraging neural networks for face detection and recognition, it can accurately identify authorized individuals and control access through a door lock mechanism. The use of flash memory allows for persistent storage of face data, making the system robust and reliable.

---

**Note:** Ensure that all the required libraries are included in your project and that the ESP32-CAM module is correctly connected and configured. Proper power supply and relay wiring are crucial for the safe operation of the door lock mechanism.
