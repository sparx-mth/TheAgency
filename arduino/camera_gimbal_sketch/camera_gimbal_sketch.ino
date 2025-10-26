#include <Servo.h>

Servo pitchServo;
Servo yawServo;

const int PITCH_PIN = 3;
const int YAW_PIN = 2;
const int SIGNAL_PIN = 7;
const int NUM_ANGLES = 4;

int yaw_angles[] = {30, 90, 130, 180};
int pitch_angles[] = {25, 30, 30, 0};

int index = 0;

void setup() {
  Serial.begin(9600);
  pitchServo.attach(PITCH_PIN);
  yawServo.attach(YAW_PIN);
  pinMode(SIGNAL_PIN, OUTPUT);
  digitalWrite(SIGNAL_PIN, HIGH); // Initially: servo is idle
  Serial.println(HIGH);
}

void loop() {
  digitalWrite(SIGNAL_PIN, LOW);
  Serial.println(LOW);
  pitchServo.write(pitch_angles[index%NUM_ANGLES]);
  yawServo.write(yaw_angles[index%NUM_ANGLES]);
  index++;
  delay(1000);
  digitalWrite(SIGNAL_PIN, HIGH); 
  Serial.println(HIGH);

  delay(20000);  // Pause before repeating
}

