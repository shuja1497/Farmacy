package com.weknownothing.farmacy.Api;

import com.weknownothing.farmacy.Models.Data;

import retrofit2.Call;
import retrofit2.http.GET;

/**
 * Created by tanya on 29/4/18.
 */

public interface Api
{
    String BASE_URL = "http://api.weatherunlocked.com/api/forecast/";
    @GET("31.6340,74.8723?app_id=dbaf1d6c&app_key=14d88196cfe4a6442c5258102ed305e2")
    Call<Data> getData();

}
