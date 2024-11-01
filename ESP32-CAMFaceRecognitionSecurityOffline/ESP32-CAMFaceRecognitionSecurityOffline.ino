#include "esp_camera.h"        // Include the ESP32 camera library
#include "esp_timer.h"         // Include the ESP32 timer library
#include "Arduino.h"           // Include the Arduino core library
#include "fd_forward.h"        // Include face detection forward declarations
#include "fr_forward.h"        // Include face recognition forward declarations
#include "fr_flash.h"          // Include face recognition flash storage functions

#define ENROLL_CONFIRM_TIMES 5 // Number of times to confirm enrollment
#define FACE_ID_SAVE_NUMBER 100 // Maximum number of faces to save

#define CAMERA_MODEL_AI_THINKER // Define the camera model as AI Thinker
#include "camera_pins.h"        // Include camera pin definitions

camera_fb_t * fb = NULL;        // Initialize the frame buffer pointer to NULL

#define relay_pin 2             // Define the relay pin (GPIO 2); pin 12 can also be used
unsigned long door_opened_millis = 0; // Variable to store the time when the door was opened
long interval = 5000;           // Duration (in milliseconds) to keep the door unlocked

void app_facenet_main();        // Function prototype for face recognition initialization

// Structure to hold image processing results
typedef struct
{
  uint8_t *image;               // Pointer to the image data
  box_array_t *net_boxes;       // Pointer to the detected face boxes
  dl_matrix3d_t *face_id;       // Pointer to the face ID matrix
} http_img_process_result;

// Function to configure the Multi-Task Cascaded Convolutional Neural Network (MTCNN)
static inline mtmn_config_t app_mtmn_config()
{
  mtmn_config_t mtmn_config = {0}; // Initialize MTCNN configuration structure
  mtmn_config.type = FAST;         // Set detection type to FAST
  mtmn_config.min_face = 80;       // Set minimum face size to detect
  mtmn_config.pyramid = 0.707;     // Set pyramid scaling factor
  mtmn_config.pyramid_times = 4;   // Set number of pyramid levels
  mtmn_config.p_threshold.score = 0.6;            // Set P-Net score threshold
  mtmn_config.p_threshold.nms = 0.7;              // Set P-Net non-maximum suppression threshold
  mtmn_config.p_threshold.candidate_number = 20;  // Set P-Net candidate number
  mtmn_config.r_threshold.score = 0.7;            // Set R-Net score threshold
  mtmn_config.r_threshold.nms = 0.7;              // Set R-Net non-maximum suppression threshold
  mtmn_config.r_threshold.candidate_number = 10;  // Set R-Net candidate number
  mtmn_config.o_threshold.score = 0.7;            // Set O-Net score threshold
  mtmn_config.o_threshold.nms = 0.7;              // Set O-Net non-maximum suppression threshold
  mtmn_config.o_threshold.candidate_number = 1;   // Set O-Net candidate number
  return mtmn_config;                             // Return the configured MTCNN settings
}
mtmn_config_t mtmn_config = app_mtmn_config();    // Initialize MTCNN configuration

face_id_name_list st_face_list;                   // Initialize the face ID list structure
static dl_matrix3du_t *aligned_face = NULL;       // Pointer for aligned face data

void setup() {
  Serial.begin(115200);        // Initialize serial communication at 115200 baud
  Serial.println();            // Print a newline to the serial monitor

  digitalWrite(relay_pin, LOW); // Ensure the relay is initially off (door locked)
  pinMode(relay_pin, OUTPUT);   // Set the relay pin as an output

  camera_config_t config;       // Create a camera configuration structure
  config.ledc_channel = LEDC_CHANNEL_0; // Set LEDC PWM channel
  config.ledc_timer = LEDC_TIMER_0;     // Set LEDC timer
  config.pin_d0 = Y2_GPIO_NUM;          // Assign camera data pins
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;      // Assign camera clock pin
  config.pin_pclk = PCLK_GPIO_NUM;      // Assign pixel clock pin
  config.pin_vsync = VSYNC_GPIO_NUM;    // Assign vertical sync pin
  config.pin_href = HREF_GPIO_NUM;      // Assign horizontal reference pin
  config.pin_sscb_sda = SIOD_GPIO_NUM;  // Assign SCCB data pin
  config.pin_sscb_scl = SIOC_GPIO_NUM;  // Assign SCCB clock pin
  config.pin_pwdn = PWDN_GPIO_NUM;      // Assign power-down pin
  config.pin_reset = RESET_GPIO_NUM;    // Assign reset pin
  config.xclk_freq_hz = 20000000;       // Set XCLK frequency to 20 MHz
  config.pixel_format = PIXFORMAT_JPEG; // Set pixel format to JPEG
  
  if (psramFound()) {                   // Check if PSRAM is available
    config.frame_size = FRAMESIZE_UXGA; // Set frame size to UXGA (1600x1200)
    config.jpeg_quality = 10;           // Set JPEG quality (lower number = higher quality)
    config.fb_count = 2;                // Use two frame buffers
  } else {
    config.frame_size = FRAMESIZE_SVGA; // Set frame size to SVGA (800x600)
    config.jpeg_quality = 12;           // Set JPEG quality
    config.fb_count = 1;                // Use one frame buffer
  }

  esp_err_t err = esp_camera_init(&config); // Initialize the camera with the configuration
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err); // Print error message if initialization fails
    return;                      // Exit the setup function
  }

  sensor_t * s = esp_camera_sensor_get(); // Get the camera sensor handle
  s->set_framesize(s, FRAMESIZE_QVGA);    // Set frame size to QVGA (320x240)

  app_facenet_main();                     // Initialize face recognition components
}

void app_facenet_main()
{
  face_id_name_init(&st_face_list, FACE_ID_SAVE_NUMBER, ENROLL_CONFIRM_TIMES); // Initialize the face ID list
  aligned_face = dl_matrix3du_alloc(1, FACE_WIDTH, FACE_HEIGHT, 3);            // Allocate memory for aligned face data
  read_face_id_from_flash_with_name(&st_face_list);                            // Load saved face IDs from flash memory
}

// Function to unlock the door
void open_door() {
  if (digitalRead(relay_pin) == LOW) {
    digitalWrite(relay_pin, HIGH); // Energize the relay to unlock the door
    Serial.println("Door Unlocked"); // Print confirmation message
    door_opened_millis = millis();   // Record the time when the door was unlocked
  }
}

void loop() {
  dl_matrix3du_t *image_matrix = dl_matrix3du_alloc(1, 320, 240, 3); // Allocate memory for the image matrix
  http_img_process_result out_res = {0};     // Initialize image processing result structure
  out_res.image = image_matrix->item;        // Set the image pointer to the image matrix data

  fb = esp_camera_fb_get();                  // Capture a frame from the camera

  if (fb)
  {
    fmt2rgb888(fb->buf, fb->len, fb->format, out_res.image); // Convert the frame buffer to RGB888 format

    out_res.net_boxes = face_detect(image_matrix, &mtmn_config); // Perform face detection on the image

    if (out_res.net_boxes)
    {
      // Align the detected face and check if alignment was successful
      if (align_face(out_res.net_boxes, image_matrix, aligned_face) == ESP_OK)
      {
        out_res.face_id = get_face_id(aligned_face); // Extract face ID from the aligned face

        if (st_face_list.count > 0)
        {
          // Recognize the face by comparing with stored face IDs
          face_id_node *f = recognize_face_with_name(&st_face_list, out_res.face_id);
          if (f)
          {
            Serial.printf("Face recognized: %s\n", f->id_name); // Print the recognized face name
            open_door();                                        // Unlock the door
          }
          else
          {
            Serial.println("Face not recognized"); // Print message if face is not recognized
          }
        }
        dl_matrix3d_free(out_res.face_id); // Free the face ID matrix memory
      }
    }
    esp_camera_fb_return(fb); // Return the frame buffer to the driver
    fb = NULL;                // Reset the frame buffer pointer
  }

  if (millis() - interval > door_opened_millis) { 
    digitalWrite(relay_pin, LOW); // De-energize the relay to lock the door after the interval
  }
  
  dl_matrix3du_free(image_matrix); // Free the image matrix memory
}
