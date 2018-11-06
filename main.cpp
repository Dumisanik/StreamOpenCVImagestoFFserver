
#include <iostream>
#include <stdio.h>

//For sleep
#include <unistd.h>

#include <thread>

#define __STDC_CONSTANT_MACROS

#ifdef _WIN32
//Windows
extern "C"
{
#include "libavformat/avformat.h"
#include "libavutil/mathematics.h"
#include "libavutil/time.h"
};
#else
//Linux...
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <libavutil/time.h>

//Newly added by me
#include <libavutil/pixdesc.h>  //For printing with av_get_pix_fmt_name()
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>

#include <
#ifdef __cplusplus
};
#endif
#endif


// OpenCV
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif



//using namespace std;
using std::cout; using std::endl; using std::cerr;


#define STREAM_FRAME_RATE 25 /* 25 images/s */
#define STREAM_PIX_FMT AV_PIX_FMT_BGR24  /* default pix_fmt */  //#AV_PIX_FMT_YUV420P  AV_PIX_FMT_BGR24
static int sws_flags = SWS_BICUBIC;

void exitApp(AVFormatContext **ofmt_ctx, AVOutputFormat **ofmt){
    int ret = 0;

    /* close output */
    if( (*ofmt_ctx) && !((*ofmt)->flags & AVFMT_NOFILE))
        avio_close((*ofmt_ctx)->pb);
    avformat_free_context(*ofmt_ctx);
    if (ret < 0 && ret != AVERROR_EOF) {
        printf( "Error occurred.\n");
    }
}


static int read_ffserver_streams(OptionsContext *o, AVFormatContext *s, const char *filename)
{
    int i, err;
    AVFormatContext *ic = avformat_alloc_context();

    ic->interrupt_callback = int_cb;
    err = avformat_open_input(&ic, filename, NULL, NULL);
    if (err < 0)
        return err;
    /* copy stream format */
    for(i=0;i<ic->nb_streams;i++) {
        AVStream *st;
        OutputStream *ost;
        AVCodec *codec;
        const char *enc_config;

        codec = avcodec_find_encoder(ic->streams[i]->codec->codec_id);
        if (!codec) {
            av_log(s, AV_LOG_ERROR, "no encoder found for codec id %i\n", ic->streams[i]->codec->codec_id);
            return AVERROR(EINVAL);
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO)
            opt_video_codec(o, "c:v", codec->name);
        ost   = new_output_stream(o, s, codec->type, -1);
        st    = ost->st;

        avcodec_get_context_defaults3(st->codec, codec);
        enc_config = av_stream_get_recommended_encoder_configuration(ic->streams[i]);
        if (enc_config) {
            AVDictionary *opts = NULL;
            av_dict_parse_string(&opts, enc_config, "=", ",", 0);
            av_opt_set_dict2(st->codec, &opts, AV_OPT_SEARCH_CHILDREN);
            av_dict_free(&opts);
        }

        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO && !ost->stream_copy)
            choose_sample_fmt(st, codec);
        else if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO && !ost->stream_copy)
            choose_pixel_fmt(st, st->codec, codec, st->codec->pix_fmt);
        avcodec_copy_context(ost->enc_ctx, st->codec);
        if (enc_config)
            av_dict_parse_string(&ost->encoder_opts, enc_config, "=", ",", 0);
    }

    avformat_close_input(&ic);
    return err;
}

int main(int argc, char* argv[]){

    av_register_all();
    //Network
    avformat_network_init();
    int dst_width = 640;
    int dst_height = 360;
    const AVRational dst_fps = {24, 1};

    //////////////////////////////////
    /// initialize OpenCV capture as input frame generator
    const char *in_filename;
    //in_filename  = "/home/dumisani/Videos/sintel.mp4";   //1280x546
    in_filename  = "/home/dumisani/Videos/big_buck_bunny.mp4";//（Input file URL）  //640x360

    cv::VideoCapture cvcap(in_filename);
    if (!cvcap.isOpened()) {
        std::cerr << "fail to open cv::VideoCapture";
        return 2;
    }


    bool isColour_ = true;
    int numChannels = 1;
    int cvColourFormat = CV_8UC1;
    if(isColour_){
        numChannels = 3;
        cvColourFormat  = CV_8UC3;
    }

    cvcap.set(cv::CAP_PROP_FRAME_WIDTH, dst_width);
    cvcap.set(cv::CAP_PROP_FRAME_HEIGHT, dst_height);
    // allocate cv::Mat with extra bytes (required by AVFrame::data)
    std::vector<uint8_t> imgbuf(dst_height * dst_width * numChannels + 16);
    cv::Mat image(dst_height, dst_width, cvColourFormat, imgbuf.data(), dst_width * 3);
    cv::Mat grayImage = cv::Mat::zeros(dst_height, dst_width, CV_8UC1);
    ///================================

    /////////////////////////////////
    /// FFMPEG
    AVOutputFormat *file_oformat = NULL;
    //Input AVFormatContext and Output AVFormatContext
    AVFormatContext *oc = NULL;
    const char *out_filename;
    int ret, i;
    int videoindex=-1;
    int frame_index=0;
    int64_t start_time=0;

    out_filename = "http://localhost:8090/feed1.ffm";

    ret = avformat_alloc_output_context2(&oc, nullptr, "ffm", out_filename);
    if (!oc) {
        printf( "Could not create output context\n");
        ret = AVERROR_UNKNOWN;
        exitApp(&oc, &file_oformat);
        return -1;
    }
    file_oformat = oc->oformat;


    ///TODO ffmpeg_opt.c line 1917

    //Read ffserver streams



    ////////////////////////////////
    /// create new video stream
    int i, err;
    AVFormatContext *ic = avformat_alloc_context();

    ic->interrupt_callback = int_cb;
    err = avformat_open_input(&ic, out_filename, NULL, NULL);
    if (err < 0)
        return err;
    /* copy stream format */
    for(i=0;i<ic->nb_streams;i++) {
        AVStream *st;
        AVCodec *codec;
        const char *enc_config;

        codec = avcodec_find_encoder(ic->streams[i]->codec->codec_id);
        if (!codec) {
            av_log(s, AV_LOG_ERROR, "no encoder found for codec id %i\n", ic->streams[i]->codec->codec_id);
            return AVERROR(EINVAL);
        }

        /*
        if (codec->type == AVMEDIA_TYPE_VIDEO)
            opt_video_codec(o, "c:v", codec->name);
        */
        st   = avformat_new_stream(oc, codec); //new_output_stream(o, s, codec->type, -1);

        avcodec_get_context_defaults3(st->codec, codec);
        enc_config = av_stream_get_recommended_encoder_configuration(ic->streams[i]);
        if (enc_config) {
            AVDictionary *opts = NULL;
            av_dict_parse_string(&opts, enc_config, "=", ",", 0);
            av_opt_set_dict2(st->codec, &opts, AV_OPT_SEARCH_CHILDREN);
            av_dict_free(&opts);
        }

        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO && !ost->stream_copy)
            choose_sample_fmt(st, codec);
        else if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO && !ost->stream_copy)
            choose_pixel_fmt(st, st->codec, codec, st->codec->pix_fmt);
        avcodec_copy_context(ost->enc_ctx, st->codec);
        if (enc_config)
            av_dict_parse_string(&ost->encoder_opts, enc_config, "=", ",", 0);
    }

    avformat_close_input(&ic);
    return err;


    /* find the encoder */
    AVCodec* outputCodec = avcodec_find_encoder(oc->oformat->video_codec);
    if (!outputCodec) {
        fprintf(stderr, "Could not find encoder for '%s'\n",
                avcodec_get_name(file_oformat->video_codec));
        exit(1);
    }

    cout<<"Codec = "<<avcodec_get_name(file_oformat->video_codec)<<endl;

    AVStream *output_stream = avformat_new_stream(oc, outputCodec);
    if (!output_stream){
        std::cerr << "fail to create an avformat_new_stream()";
        return 2;
    }
    avcodec_get_context_defaults3(output_stream->codec, outputCodec);

    //============================================================
    videoindex = 0;

    //Create output AVStream according to input AVStream
    output_stream->codec->width = dst_width;
    output_stream->codec->height = dst_height;
    output_stream->codec->pix_fmt = outputCodec->pix_fmts[0];
    output_stream->codec->time_base = output_stream->time_base = av_inv_q(dst_fps);
    output_stream->r_frame_rate = output_stream->avg_frame_rate = dst_fps;

    //# Try this for improving the quality of the decoded video
    output_stream->codec->bit_rate = 200000*6;


    //
    av_dict_set(&oc->metadata, "analyzeduration", "1000", 0);
    
    


    if (!output_stream) {
        printf( "Failed allocating output stream\n");
        ret = AVERROR_UNKNOWN;
        exitApp(&oc, &file_oformat);
        return -1;
    }

    output_stream->codec->codec_tag = 0;
    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        output_stream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;


    // open video encoder
    ret = avcodec_open2(output_stream->codec, outputCodec, nullptr);
    if (ret < 0) {
        std::cerr << "fail to avcodec_open2: ret=" << ret;
        return 2;
    }

    //Dump Format------------------
    av_dump_format(oc, 0, out_filename, 1);
    //Open output URL
    if (!(file_oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&oc->pb, out_filename, AVIO_FLAG_WRITE);
        if (ret < 0) {
            printf( "Could not open output URL '%s' ", out_filename);
            exitApp(&oc, &file_oformat);
            return -1;
        }
    }


    /////////////////////////////////
    /// AV Frame
    // initialize sample scaler
    //static struct SwsContext *sws_ctx;
    SwsContext* sws_ctx = sws_getCachedContext(
        nullptr, dst_width, dst_height, STREAM_PIX_FMT,
        dst_width, dst_height, output_stream->codec->pix_fmt, SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!sws_ctx){
        std::cerr << "fail to sws_getCachedContext";
        return 2;
    }

    
    // allocate frame buffer for encoding
    AVFrame *frame = av_frame_alloc();

    std::vector<uint8_t> framebuf(avpicture_get_size(output_stream->codec->pix_fmt, dst_width, dst_height));
    avpicture_fill(reinterpret_cast<AVPicture*>(frame), framebuf.data(), output_stream->codec->pix_fmt, dst_width, dst_height);
    frame->width = dst_width;
    frame->height = dst_height;
    frame->format = static_cast<int>(output_stream->codec->pix_fmt);

    std::cout<< "format:  " << oc->oformat->name << "\n"
           << "vcodec:  " << outputCodec->name  << "\n"
           << "size:    " << dst_width << 'x' << dst_height << "\n"
           << "fps:     " << av_q2d(oc->streams[videoindex]->avg_frame_rate) << "\n"
           << "pixfmt:  " << av_get_pix_fmt_name(oc->streams[videoindex]->codec->pix_fmt) << "\n"
           << "bit_rate: "<< oc->streams[videoindex]->codec->bit_rate  << "\n"
           << "time_base: num=" << oc->streams[videoindex]->time_base.num
           << " denomenator=" << oc->streams[videoindex]->time_base.den << "\n"
           << std::flush;
    ///==================================


    std::cout<<"Waiting for client to connect..." <<std::endl;
    int connectionRetries = 1500;

    ret = -1;


    ret = avformat_write_header(oc, NULL);

    std::cout<<"Starting app.." <<std::endl;



    start_time=av_gettime();
    bool end_of_stream = false;
    int got_pkt = 0;

    while (!end_of_stream) {


        std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

        if (!end_of_stream) {
            // retrieve source image
            cvcap >> image;
            //cv::imshow("press ESC to exit", image);

            int pressedKey = cv::waitKey(33);
            if (pressedKey == 0x1b){
                end_of_stream = true;
                break;
            }
        }

        // convert cv::Mat(OpenCV) to AVFrame(FFmpeg)
        if(image.type()==CV_8UC3 && !isColour_){
            //Convert image to grayscale
            cv::cvtColor(image, grayImage, CV_BGR2GRAY);
            const int stride[] = { static_cast<int>(grayImage.step[0]) };
            sws_scale(sws_ctx, &grayImage.data, stride, 0, grayImage.rows, frame->data, frame->linesize);
            frame->pts = frame_index++;
        }else{
            const int stride[] = { static_cast<int>(image.step[0]) };
            sws_scale(sws_ctx, &image.data, stride, 0, image.rows, frame->data, frame->linesize);
            frame->pts = frame_index++;
        }


        sws_ctx = sws_getContext(dst_width, dst_height, STREAM_PIX_FMT,    //AV_PIX_FMT_YUV420P  AV_PIX_FMT_RGB24
                                 dst_width, dst_height, output_stream->codec->pix_fmt,
                                 sws_flags, NULL, NULL, NULL);

        if (!sws_ctx){
            fprintf(stderr, "Could not initialize the conversion context\n");
            exit(1);
        }



        AVPacket pkt;

        //If the video is not a Raw video)";
        if (!(oc->oformat->flags & AVFMT_RAWPICTURE)) {
            // encode video frame
            pkt.data = nullptr;
            pkt.size = 0;
            av_init_packet(&pkt);
            ret = avcodec_encode_video2(output_stream->codec, &pkt, end_of_stream ? nullptr : frame, &got_pkt);
        }

        if (ret < 0) {
            std::cerr << "fail to avcodec_encode_video2: ret=" << ret << "\n";
            break;
        }
        if (got_pkt){
            // rescale packet timestamp
            pkt.duration = 1;
            av_packet_rescale_ts(&pkt, output_stream->codec->time_base, output_stream->time_base);

            // write packet

        }


        AVStream *out_stream;
        //Get an AVPacket
        /*
        ret = av_read_frame(ofmt_ctx, &pkt);
        if (ret < 0)
            break;
        */


        //FIX：No PTS (Example: Raw H.264)
        //Simple Write PTS
        if(pkt.pts==AV_NOPTS_VALUE){
            //Write PTS
            AVRational time_base1=oc->streams[videoindex]->time_base;
            //Duration between 2 frames (us)
            int64_t calc_duration=(double)AV_TIME_BASE/av_q2d(oc->streams[videoindex]->r_frame_rate); //# Added x2 to improve stream fps
            //Parameters
            pkt.pts=(double)(frame_index*calc_duration)/(double)(av_q2d(time_base1)*AV_TIME_BASE);
            pkt.dts=pkt.pts;
            pkt.duration=(double)calc_duration/(double)(av_q2d(time_base1)*AV_TIME_BASE);
        }
        //Important:Delay
        if(pkt.stream_index==videoindex){
            AVRational time_base=oc->streams[videoindex]->time_base;
            AVRational time_base_q={1,AV_TIME_BASE};
            int64_t pts_time = av_rescale_q(pkt.dts, time_base, time_base_q);
            int64_t now_time = av_gettime() - start_time;
            if (pts_time > now_time)
                av_usleep(pts_time - now_time * int64_t(2));
        }




        out_stream = oc->streams[pkt.stream_index];
        /* copy packet */
        //Convert PTS/DTS

        //Print to Screen
        if(pkt.stream_index==videoindex){
            if(frame_index==0)
                printf("Streaming frames to %s\n", out_filename);

            //printf("Sent %8d video frames to output URL\n",frame_index);
            frame_index++;
        }

        ///Write a packet to an output media file ensuring correct interleaving.
        //ret = av_write_frame(ofmt_ctx, &pkt);
        ret = av_interleaved_write_frame(oc, &pkt);

        if (ret < 0) {
            printf( "Error muxing packet\n");
            break;
        }


        av_free_packet(&pkt);

        std::chrono::time_point<std::chrono::system_clock> endTime = std::chrono::system_clock::now();
        //Convert elapsed time to seconds
        std::chrono::duration<double, std::milli> timeElapsed = endTime - startTime;
        int elapsedSeconds = timeElapsed.count() / 1000;
        std::cout<<"Elapsed time (milliseconds) per frame: "<< timeElapsed.count() <<endl;


    }
    //Write file trailer
    av_write_trailer(oc);


    return 0;
}



