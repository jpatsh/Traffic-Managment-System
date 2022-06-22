// Authored by Jacob Patshkowski and Wilson Ibyishaka
//EENG 490 Eastern Washington University 

#include <ArduinoWebsockets.h>
#include <ESP8266WiFi.h>

const char* ssid = "Jacob"; //Enter SSID
const char* password = "Patshkowski"; //Enter Password
int ledState = LOW;

using namespace websockets;

WebsocketsServer server;
void setup() {

  //setting up the lights
  pinMode(D0,OUTPUT);     //RED LIGHT
  pinMode(D1,OUTPUT);    //YELLOW LIGHT
  pinMode(D2,OUTPUT);    //GREEN LIGHT
  pinMode(D3,OUTPUT);    //R/Red walking
  pinMode(D4,OUTPUT);    //R/white Walking
  pinMode(D5,OUTPUT);    //L/red Walking
  pinMode(D6,OUTPUT);   // L/white walking
  pinMode(D7,INPUT);    //R/Button
  pinMode(D8,INPUT);    //L/Button

  
  Serial.begin(115200);
  
  
  // Connect to wifi
  WiFi.begin(ssid, password);

  // Wait some time to connect to wifi
  for(int i = 0; i < 15 && WiFi.status() != WL_CONNECTED; i++) {
      Serial.print(".");
      delay(1000);
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());   //You can get IP address assigned to ESP

  server.listen(8765);
  Serial.print("Is server live? ");
  Serial.println(server.available());
}

void loop() {
  String msgg = "";
  
  WebsocketsClient client = server.accept();
  while(client.available()) {
    
    if (digitalRead(D8) == HIGH){
      client.send("1");
    }
    if (digitalRead(D7) == HIGH){
      client.send("2");
    } 
    if ((digitalRead(D7) == LOW) && (digitalRead(D8) == LOW)) {
      client.send("0");
    }
    
    WebsocketsMessage msg = client.readBlocking();
    Serial.println(msg.data());
    msgg = String(msg.data());
    if(msgg.charAt(2) == '1'){
       digitalWrite(D2, HIGH);
    }else{
      digitalWrite(D2, LOW);
    }
    if(msgg.charAt(1) == '1'){
       digitalWrite(D1, HIGH);
    }else{
      digitalWrite(D1, LOW);
    }
    if(msgg.charAt(0) == '1'){
       digitalWrite(D0, HIGH);
    }else{
      digitalWrite(D0, LOW);      
    }
    if(msgg.charAt(3) == '1'){
       digitalWrite(D3, HIGH);
    }else{
      digitalWrite(D3, LOW);
    }
    if(msgg.charAt(4) == '1'){
       digitalWrite(D4, HIGH);
    }else{
      digitalWrite(D4, LOW);
    }
    if(msgg.charAt(5) == '1'){
       digitalWrite(D5, HIGH);
    }else{
      digitalWrite(D5, LOW);      
    }
    if(msgg.charAt(6) == '1'){
       digitalWrite(D6, HIGH);
    }else{
      digitalWrite(D6, LOW);      
    }
        
          
      
    
   
    
      
   
  }
  
 delay(1000);
}
