package com.example.exceptions;

public class StaleElementException extends RuntimeException {
    public StaleElementException(String locator) {
        super("Element is no longer attached to the DOM: " + locator);
    }
}
