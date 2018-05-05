package com.weknownothing.farmacy.Functionalities;

import android.Manifest;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.annotation.StringRes;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v4.app.DialogFragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Base64;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.Response;
import com.android.volley.RetryPolicy;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.google.android.cameraview.AspectRatio;
import com.google.android.cameraview.CameraView;
import com.weknownothing.farmacy.Utilities.AspectRatioFragment;
import com.weknownothing.farmacy.Utilities.MySingelton;
import com.weknownothing.farmacy.R;

import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;


public class DetectDiseaseActivity extends AppCompatActivity implements
        ActivityCompat.OnRequestPermissionsResultCallback,
        AspectRatioFragment.Listener {

    private static final String TAG = "MainActivity";

    private static final int REQUEST_CAMERA_PERMISSION = 1;

    private static final String FRAGMENT_DIALOG = "dialog";

    private static final int[] FLASH_OPTIONS = {
            CameraView.FLASH_AUTO,
            CameraView.FLASH_OFF,
            CameraView.FLASH_ON,
            };

    private static final int[] FLASH_ICONS = {
            R.drawable.ic_flash_auto,
            R.drawable.ic_flash_off,
            R.drawable.ic_flash_on,
            };

    private static final int[] FLASH_TITLES = {
            R.string.flash_auto,
            R.string.flash_off,
            R.string.flash_on,
            };
    public static final int INPUT_SIZE = 600;
    public static final String Uploadurl = "https://3e78b515.ngrok.io/";

    private int mCurrentFlash;

    private CameraView mCameraView;

    private Handler mBackgroundHandler;

    private View.OnClickListener mOnClickListener = new View.OnClickListener() {

        @Override
        public void onClick(View v) {
            switch (v.getId()) {
            case R.id.take_picture:
            if (mCameraView != null) {
            mCameraView.takePicture();
            }
            break;
            }
        }
    };
        private Bitmap mCompressedBitmap;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detect_disease);

        mCameraView = findViewById(R.id.camera);
        if (mCameraView != null) {
        mCameraView.addCallback(mCallback);
        }
        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.take_picture);
        if (fab != null) {
        fab.setOnClickListener(mOnClickListener);
        }
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
        actionBar.setDisplayShowTitleEnabled(false);
        }

    }

    @Override
    protected void onResume() {
            super.onResume();
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            mCameraView.start();
            }
            else if (ActivityCompat.shouldShowRequestPermissionRationale(this,
            Manifest.permission.CAMERA)) {
                ConfirmationDialogFragment
                .newInstance(R.string.camera_permission_confirmation,
                new String[]{Manifest.permission.CAMERA},
                REQUEST_CAMERA_PERMISSION,
                R.string.camera_permission_not_granted)
                .show(getSupportFragmentManager(), FRAGMENT_DIALOG);
            } else {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},
                REQUEST_CAMERA_PERMISSION);
            }
    }

@Override
protected void onPause() {
        mCameraView.stop();
        super.onPause();
        }

@Override
protected void onDestroy() {
        super.onDestroy();
        if (mBackgroundHandler != null) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
        mBackgroundHandler.getLooper().quitSafely();
        } else {
        mBackgroundHandler.getLooper().quit();
        }
        mBackgroundHandler = null;
        }
        }

@Override
public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
@NonNull int[] grantResults) {
        switch (requestCode) {
        case REQUEST_CAMERA_PERMISSION:
        if (permissions.length != 1 || grantResults.length != 1) {
        throw new RuntimeException("Error on requesting camera permission.");
        }
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
        Toast.makeText(this, R.string.camera_permission_not_granted,
        Toast.LENGTH_SHORT).show();
        }
        // No need to start camera here; it is handled by onResume
        break;
        }
        }

@Override
public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
        }

@Override
public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
        case R.id.aspect_ratio:
        FragmentManager fragmentManager = getSupportFragmentManager();
        if (mCameraView != null
        && fragmentManager.findFragmentByTag(FRAGMENT_DIALOG) == null) {
final Set<AspectRatio> ratios = mCameraView.getSupportedAspectRatios();
final AspectRatio currentRatio = mCameraView.getAspectRatio();
        AspectRatioFragment.newInstance(ratios, currentRatio)
        .show(fragmentManager, FRAGMENT_DIALOG);
        }
        return true;
        case R.id.switch_flash:
        if (mCameraView != null) {
        mCurrentFlash = (mCurrentFlash + 1) % FLASH_OPTIONS.length;
        item.setTitle(FLASH_TITLES[mCurrentFlash]);
        item.setIcon(FLASH_ICONS[mCurrentFlash]);
        mCameraView.setFlash(FLASH_OPTIONS[mCurrentFlash]);
        }
        return true;
        case R.id.switch_camera:
        if (mCameraView != null) {
        int facing = mCameraView.getFacing();
        mCameraView.setFacing(facing == CameraView.FACING_FRONT ?
        CameraView.FACING_BACK : CameraView.FACING_FRONT);
        }
        return true;
        }
        return super.onOptionsItemSelected(item);
        }

@Override
public void onAspectRatioSelected(@NonNull AspectRatio ratio) {
        if (mCameraView != null) {
        Toast.makeText(this, ratio.toString(), Toast.LENGTH_SHORT).show();
        mCameraView.setAspectRatio(ratio);
        }
        }

private Handler getBackgroundHandler() {
        if (mBackgroundHandler == null) {
        HandlerThread thread = new HandlerThread("background");
        thread.start();
        mBackgroundHandler = new Handler(thread.getLooper());
        }
        return mBackgroundHandler;
        }

private CameraView.Callback mCallback
        = new CameraView.Callback() {

        @Override
        public void onCameraOpened(CameraView cameraView) {
                Log.d(TAG, "onCameraOpened");
                }

        @Override
        public void onCameraClosed(CameraView cameraView) {
                Log.d(TAG, "onCameraClosed");
                }
        @Override
        public void onPictureTaken(CameraView cameraView, final byte[] picture) {
                Log.d(TAG, "onPictureTaken " + picture.length);
                Toast.makeText(cameraView.getContext(), R.string.picture_taken, Toast.LENGTH_SHORT)
                .show();
            mCompressedBitmap = getBitmap(picture);
            UploadImage upld = new UploadImage();
            upld.execute();
        }
    };

    private Bitmap getBitmap(byte[] picture) {
        Log.d(TAG, "onPictureTaken: "+ Arrays.toString(picture));

        Bitmap bitmap = BitmapFactory.decodeByteArray(picture, 0, picture.length);

        Log.d(TAG, "onPictureTaken: widthhhhh---------------"+bitmap.getWidth());

        bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

        Log.d(TAG, "onPictureTaken: newww  widthhhhh---------------"+bitmap.getWidth());

        //bitmap.setPixel(320,320,0);
        Log.d(TAG, "onPictureTaken: newww  widthhhhh---------------"+bitmap.getWidth());

        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);

        return bitmap;
    }

    private class UploadImage extends AsyncTask<Void,Void,Void> {

        @Override
        protected Void doInBackground(Void... params) {
//            String.format(Uploadurl,ImagetoString(bitmap));
            StringRequest stringRequest = new StringRequest(Request.Method.POST, Uploadurl,
                    new Response.Listener<String>() {
                        @Override
                        public void onResponse(String response) {
                            Log.d(TAG, "*******************onResponse: "+response);

                            Toast.makeText(getApplicationContext(),response,Toast.LENGTH_LONG).show();
                        }
                    }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Log.d(TAG, "onErrorResponse: "+ error);

                }
            })
            {
                @Override
                protected Map<String, String> getParams() throws AuthFailureError {
                    Log.d(TAG, "getParams: herererrerererre");
                    Map<String,String> params =new HashMap<>();
                    params.put("image",ImagetoString(mCompressedBitmap));

                    return params;
                }
            };
            RetryPolicy policy = new DefaultRetryPolicy(50000, DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                    DefaultRetryPolicy.DEFAULT_BACKOFF_MULT);
            stringRequest.setRetryPolicy(policy);

            MySingelton.getInstance(DetectDiseaseActivity.this).addToRequestQue(stringRequest);

            return null;
        }
    }

    private String ImagetoString(Bitmap bitmap)
    {
        Log.d(TAG, "ImagetoString: in");
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG,100,byteArrayOutputStream);
        byte[] imgBytes = byteArrayOutputStream.toByteArray();
        Log.d(TAG, "ImagetoString: out*******"+imgBytes.length);
        return Base64.encodeToString(imgBytes,Base64.DEFAULT);
    }


    public static class ConfirmationDialogFragment extends DialogFragment {

    private static final String ARG_MESSAGE = "message";
    private static final String ARG_PERMISSIONS = "permissions";
    private static final String ARG_REQUEST_CODE = "request_code";
    private static final String ARG_NOT_GRANTED_MESSAGE = "not_granted_message";

    public static ConfirmationDialogFragment newInstance(@StringRes int message,
                                                         String[] permissions, int requestCode, @StringRes int notGrantedMessage) {
        ConfirmationDialogFragment fragment = new ConfirmationDialogFragment();
        Bundle args = new Bundle();
        args.putInt(ARG_MESSAGE, message);
        args.putStringArray(ARG_PERMISSIONS, permissions);
        args.putInt(ARG_REQUEST_CODE, requestCode);
        args.putInt(ARG_NOT_GRANTED_MESSAGE, notGrantedMessage);
        fragment.setArguments(args);
        return fragment;
    }

    @NonNull
    @Override
    public Dialog onCreateDialog(Bundle savedInstanceState) {
        final Bundle args = getArguments();
        return new AlertDialog.Builder(getActivity())
                .setMessage(args.getInt(ARG_MESSAGE))
                .setPositiveButton(android.R.string.ok,
                        new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                String[] permissions = args.getStringArray(ARG_PERMISSIONS);
                                if (permissions == null) {
                                    throw new IllegalArgumentException();
                                }
                                ActivityCompat.requestPermissions(getActivity(),
                                        permissions, args.getInt(ARG_REQUEST_CODE));
                            }
                        })
                .setNegativeButton(android.R.string.cancel,
                        new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                Toast.makeText(getActivity(),
                                        args.getInt(ARG_NOT_GRANTED_MESSAGE),
                                        Toast.LENGTH_SHORT).show();
                            }
                        })
                .create();
    }
    }
}

