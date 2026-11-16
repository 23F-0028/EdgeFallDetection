
#include <Wire.h>

#define SDA_PIN 8
#define SCL_PIN 9

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println("MPU6050 RAW REGISTER TEST");

  // Read WHO_AM_I register
  Wire.beginTransmission(0x68);
  Wire.write(0x75);
  byte error = Wire.endTransmission(false);

  if (error != 0) {
    Serial.println("Register address transmission failed!");
    return;
  }

  Wire.requestFrom(0x68, 1);

  if (Wire.available()) {
    byte identity = Wire.read();

    Serial.print("WHO_AM_I: 0x");
    Serial.println(identity, HEX);

    if (identity == 0x68) {
      Serial.println("MPU6050 CHIP IDENTIFIED SUCCESSFULLY");
    } else {
      Serial.println("Unexpected chip identity");
    }
  } else {
    Serial.println("Could not read WHO_AM_I register");
  }
}

void loop() {
}
