package com.weknownothing.farmacy.Utilities;

import android.content.Context;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.weknownothing.farmacy.Models.Days;
import com.weknownothing.farmacy.R;

import java.util.List;

/**
 * Created by tanya on 1/5/18.
 */


public class MyAdapter extends RecyclerView.Adapter<MyAdapter.ViewHolder> {

    private List<Days> daysList;
    private Context context;

    public MyAdapter(List<Days> daysList, Context context) {
        this.daysList = daysList;
        this.context = context;
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {

        View v = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.list_item , parent , false);

        return new ViewHolder(v);
    }

    @Override
    public void onBindViewHolder(ViewHolder holder, int position) {

        Days listItem = daysList.get(position);

        holder.datetxt.setText(listItem.getDate());
        holder.mintv.setText(String.valueOf(listItem.getTemp_min_c()));
        holder.maxtv.setText(String.valueOf(listItem.getTemp_max_c()));
        holder.minhv.setText(String.valueOf(listItem.getHumid_min_pct()));
        holder.maxhv.setText(String.valueOf(listItem.getHumid_max_pct()));
        holder.precv.setText(String.valueOf(listItem.getPrecip_total_mm()));


    }

    @Override
    public int getItemCount() {
        return daysList.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {

        private TextView datetxt, mintv, maxtv, minhv, maxhv, precv;

        private LinearLayout linearLayout;

        public ViewHolder(View itemView) {
            super(itemView);

            datetxt = (TextView)itemView.findViewById(R.id.datetxt);
            mintv = (TextView)itemView.findViewById(R.id.mintv);
            maxtv = (TextView)itemView.findViewById(R.id.maxtv);
            minhv = (TextView)itemView.findViewById(R.id.minhv);
            maxhv = (TextView)itemView.findViewById(R.id.maxhv);
            precv = (TextView)itemView.findViewById(R.id.precv);

           // linearLayout = (LinearLayout)itemView.findViewById(R.id.lin);


        }
    }
}
