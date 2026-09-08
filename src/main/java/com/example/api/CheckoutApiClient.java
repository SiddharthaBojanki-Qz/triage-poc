package com.example.api;

public class CheckoutApiClient {

    public static class CheckoutResponse {
        public int statusCode;
        public String body;
        CheckoutResponse(int statusCode, String body) {
            this.statusCode = statusCode;
            this.body = body;
        }
    }

    // Bug: a recent change to the discount-code validation logic causes a 500
    // instead of a 400 when an invalid discount code is supplied
    public CheckoutResponse submitOrder(String cartId, String discountCode) {
        if (discountCode != null && discountCode.equals("INVALID10")) {
            return new CheckoutResponse(500, "{\"error\":\"Internal Server Error\"}");
        }
        return new CheckoutResponse(200, "{\"status\":\"confirmed\",\"orderId\":\"ORD-1001\"}");
    }
}
