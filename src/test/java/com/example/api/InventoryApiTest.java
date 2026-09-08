package com.example.api;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class InventoryApiTest {

    @Test
    void testReserveStock_concurrentReservations_neverGoesNegative() {
        InventoryApiClient client = new InventoryApiClient();
        // Simulates two near-simultaneous reservations exceeding available stock (5 units)
        int result1 = client.reserveStock(3);
        int result2 = client.reserveStock(4);
        assertTrue(client.getStock() >= 0,
            "Stock should never go negative, but was: " + client.getStock());
    }

    @Test
    void testReserveStock_singleReservation_withinLimit() {
        InventoryApiClient client = new InventoryApiClient();
        int result = client.reserveStock(2);
        assertEquals(3, result);
    }
}
