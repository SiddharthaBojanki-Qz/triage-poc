package com.example.exceptions;

public class ElementNotFoundException extends RuntimeException {
    public ElementNotFoundException(String locator) {
        super("Unable to locate element: " + locator + " (no such element)");
    }
}
