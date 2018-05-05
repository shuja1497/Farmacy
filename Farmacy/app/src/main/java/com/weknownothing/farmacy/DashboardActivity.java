package com.weknownothing.farmacy;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

import com.weknownothing.farmacy.Functionalities.DetectDiseaseActivity;
import com.weknownothing.farmacy.Functionalities.WeatherForecastActivity;

public class DashboardActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dashboard);
    }

    public void onClick(View view){

        switch (view.getId()){
            case R.id.imageView_detect_disease:
                startActivity(new Intent(this, DetectDiseaseActivity.class));
                break;

            case R.id.imageView_crop_info:
                break;

            case R.id.imageView_crop_suggestion:
                break;

            case R.id.imageView_weather_forecast:
                startActivity(new Intent(this, WeatherForecastActivity.class));
                break;

        }
    }
}
