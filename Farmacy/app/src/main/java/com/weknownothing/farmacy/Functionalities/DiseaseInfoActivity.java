package com.weknownothing.farmacy.Functionalities;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;

import com.weknownothing.farmacy.R;

import java.lang.reflect.Array;
import java.util.ArrayList;

public class DiseaseInfoActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_disease_info);

        String[] arrayListDisease = this.getResources().getStringArray(R.array.diseases);
        String[] arrayListCauses = this.getResources().getStringArray(R.array.causes);
        String[] arrayListTreatment = this.getResources().getStringArray(R.array.treatment);

        String index = getIntent().getExtras().getString("index","No disease");

        Toast.makeText(this, "index is "+ index, Toast.LENGTH_SHORT).show();

        TextView tv_disease = findViewById(R.id.textView_disease);
        TextView tv_causes = findViewById(R.id.textView_causes);
        TextView tv_treatment = findViewById(R.id.textView_treatment);

        tv_disease.setText(arrayListDisease[Integer.parseInt(index)]);
        tv_causes.setText(arrayListCauses[Integer.parseInt(index)]);
        tv_treatment.setText(arrayListTreatment[Integer.parseInt(index)]);
    }
}
