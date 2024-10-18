package com.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
@RestController
public class App {

    public static void main(String[] args) {
        SpringApplication.run(App.class, args);
        System.out.println("Server is ready to take orders!");
    }

    @GetMapping("/hello")
    public String sayHello() {
        System.out.println("Order received: Hello World!");
        return "Hello World!";
    }

    @GetMapping("/helloAmul")
    public String sayHelloAmul() {
        System.out.println("Order received: Hello Amul, see this works differently!");
        return "Hello Amul - see this website is working differently!";
    }
}
