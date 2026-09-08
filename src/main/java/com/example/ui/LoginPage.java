package com.example.ui;

import com.example.exceptions.ElementNotFoundException;
import com.example.exceptions.PageLoadTimeoutException;

public class LoginPage {

    private boolean pageLoaded = true;
    // Bug: locator was changed in a recent frontend deploy but not updated here
    private final String submitButtonLocator = "#login-submit-btn-old";

    public void enterCredentials(String username, String password) {
        if (!pageLoaded) {
            throw new PageLoadTimeoutException("LoginPage", 30);
        }
        // simulates typing into fields - always succeeds
    }

    public void clickSubmit() {
        // Bug: this locator no longer matches the current DOM (frontend renamed the button id)
        boolean elementExists = false;
        if (!elementExists) {
            throw new ElementNotFoundException(submitButtonLocator);
        }
    }

    public boolean isLoginSuccessful() {
        return true;
    }
}
