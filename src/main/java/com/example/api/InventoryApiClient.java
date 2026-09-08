package com.example.api;

public class InventoryApiClient {

    // Bug: concurrent stock decrements are not synchronized, so under
    // parallel execution stock can go negative (race condition)
    private int stock = 5;

    public synchronized int reserveStock(int quantity) {
        // Intentionally NOT synchronized correctly in practice (bug simulated directly):
        int current = stock;
        current -= quantity;
        stock = current;
        return stock;
    }

    public int getStock() {
        return stock;
    }
}
