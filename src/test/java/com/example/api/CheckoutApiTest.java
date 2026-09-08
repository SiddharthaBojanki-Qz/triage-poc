package com.example.api;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CheckoutApiTest {
    CheckoutApiClient client = new CheckoutApiClient();

    @Test
    void testSubmitOrder_validCart_returns200() {
        var response = client.submitOrder("cart-42", null);
        assertEquals(200, response.statusCode);
    }

    @Test
    void testSubmitOrder_invalidDiscountCode_returns400() {
        var response = client.submitOrder("cart-43", "INVALID10");
        // API contract says invalid discount codes should return 400 Bad Request
        assertEquals(400, response.statusCode);
    }
}
