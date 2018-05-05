package com.weknownothing.farmacy.Models;
/**
 * Created by tanya on 29/4/18.
 */
public class Days
{
    private String date;
    private float temp_min_c;
    private float temp_max_c;
    private float humid_min_pct;
    private float humid_max_pct;
    private float precip_total_mm;

    public Days(String date, float temp_min_c, float temp_max_c, float humid_min_pct, float humid_max_pct, float precip_total_mm) {
        this.date = date;
        this.temp_min_c = temp_min_c;
        this.temp_max_c = temp_max_c;
        this.humid_min_pct = humid_min_pct;
        this.humid_max_pct = humid_max_pct;
        this.precip_total_mm = precip_total_mm;
    }

    public String getDate() {
        return date;
    }

    public float getTemp_min_c() {
        return temp_min_c;
    }

    public float getTemp_max_c() {
        return temp_max_c;
    }

    public float getHumid_min_pct() {
        return humid_min_pct;
    }

    public float getHumid_max_pct() {
        return humid_max_pct;
    }

    public float getPrecip_total_mm() {
        return precip_total_mm;
    }
}