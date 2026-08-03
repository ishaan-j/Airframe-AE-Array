const int MIC1 = A0;
const int MIC2 = A1;
const int THRESHOLD = 40;
const int BASELINE = 512;
long lastEvent = 0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int v1 = abs(analogRead(MIC1) - BASELINE);
  int v2 = abs(analogRead(MIC2) - BASELINE);
  
  if ((v1 > THRESHOLD || v2 > THRESHOLD) 
       && millis() - lastEvent > 200) {
    Serial.print(millis());
    Serial.print(",");
    Serial.print(v1);
    Serial.print(",");
    Serial.println(v2);
    lastEvent = millis();
  }
}
