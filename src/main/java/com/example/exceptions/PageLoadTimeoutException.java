package com.example.exceptions;

public class PageLoadTimeoutException extends RuntimeException {
    public PageLoadTimeoutException(String pageName, int timeoutSeconds) {
        super("Timed out after " + timeoutSeconds + "s waiting for page to load: " + pageName);
    }
}
