package com.example.ui;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LoginPageTest {
    LoginPage loginPage = new LoginPage();

    @Test
    void testEnterCredentials_validUser() {
        assertDoesNotThrow(() -> loginPage.enterCredentials("qa_user01", "P@ssw0rd123"));
    }

    @Test
    void testClickSubmit_afterValidLogin() {
        loginPage.enterCredentials("qa_user01", "P@ssw0rd123");
        loginPage.clickSubmit();
        assertTrue(loginPage.isLoginSuccessful());
    }
}
