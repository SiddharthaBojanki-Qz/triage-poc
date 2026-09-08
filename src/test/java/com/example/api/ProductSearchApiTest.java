package com.example.api;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ProductSearchApiTest {
    ProductSearchApiClient client = new ProductSearchApiClient();

    @Test
    void testSearch_knownProduct_returnsResults() {
        var results = client.search("laptop");
        assertFalse(results.isEmpty());
    }

    @Test
    void testSearch_unknownProduct_returnsEmpty() {
        var results = client.search("xyz-unknown-item");
        assertTrue(results.isEmpty());
    }
}
