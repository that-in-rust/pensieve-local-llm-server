package com.example.helloworld;

import org.springframework.boot.SpringApplication;
   import org.springframework.boot.autoconfigure.SpringBootApplication;
   import org.springframework.web.bind.annotation.GetMapping;
   import org.springframework.web.bind.annotation.RestController;

   @SpringBootApplication
   @RestController
   public class HelloWorldApplication {

       public static void main(String[] args) {
           SpringApplication.run(HelloWorldApplication.class, args);
           System.out.println("Server is ready to take orders!");
       }

       @GetMapping("/hello")
       public String sayHello() {
           System.out.println("Order received: Hello World!");
           return "Hello World!";
       }
   }
