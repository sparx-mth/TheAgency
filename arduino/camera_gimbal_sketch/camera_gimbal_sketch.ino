#include <Servo.h>

Servo pitchServo;
Servo yawServo;

const int PITCH_PIN = 3;
const int YAW_PIN = 2;
const int SIGNAL_PIN = 7;
const int BIT0_INDEX_PIN = 4;   // LSB
const int BIT1_INDEX_PIN = 5;
const int BIT2_INDEX_PIN = 6;   // MSB


const int yaw_angles[] = {30, 90, 130, 180};
const int pitch_angles[] = {25, 30, 30, 0};
const int num_angles = sizeof(yaw_angles) / sizeof(yaw_angles[0]);

const int ANGLE_DELAY = 20000;

int index = 0;



void write_to_pins(int number) {
  const int bit0 = number & 1;
  const int bit1 = (number >> 1) & 1;
  const int bit2 = (number >> 2) & 1;
  Serial.print("Index bits (MSB first): ");
  Serial.print(bit2);
  Serial.print(bit1);
  Serial.println(bit0);
  digitalWrite(BIT0_INDEX_PIN, bit0);        
  digitalWrite(BIT1_INDEX_PIN, bit1); 
  digitalWrite(BIT2_INDEX_PIN, bit2); 
}



void setup() {
  Serial.begin(9600);
  pitchServo.attach(PITCH_PIN);
  yawServo.attach(YAW_PIN);
  pinMode(SIGNAL_PIN, OUTPUT);
  digitalWrite(SIGNAL_PIN, HIGH); // Initially: servo is idle
  Serial.println(HIGH);
}

void loop() {
  Serial.println("*****************  loop ***************");
  digitalWrite(SIGNAL_PIN, LOW);
  Serial.println("Trigger LOW. Do not take frame");

  Serial.println("Update Servo Angles");
  const int pitch = pitch_angles[index%num_angles];
  pitchServo.write(pitch);
  const int yaw = yaw_angles[index%num_angles];
  yawServo.write(yaw);
  
  Serial.print("Servo Angles (pitch, yaw): ");
  Serial.print(pitch);
  Serial.print(",");
  Serial.println(yaw);
  
  
  write_to_pins(index%num_angles);
  Serial.print("Index written to pins: ");
  Serial.println(index%num_angles);
  index++;
  digitalWrite(SIGNAL_PIN, HIGH); 
  Serial.println("Trigger HIGH. May take frame");

  delay(ANGLE_DELAY);  // Pause before repeating
}

