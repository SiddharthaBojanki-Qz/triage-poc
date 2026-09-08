package com.example.api;

import java.util.List;
import java.util.Arrays;

public class ProductSearchApiClient {

    public List<String> search(String query) {
        // Works correctly - healthy module
        if (query.equalsIgnoreCase("laptop")) {
            return Arrays.asList("Laptop Pro 15", "Laptop Air 13");
        }
        return Arrays.asList();
    }
}
