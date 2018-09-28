/*
 * Convert from OpenCV image and write movie with FFmpeg
 *
 * Usage, first start ffplay:
 * $ ffplay-rtsp_flags listen -i rtsp://127.0.0.1:8554/live.sdp
 * then run this app.
*/
#include <iostream>
#include <vector>
// FFmpeg
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

// OpenCV
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "streaming.h"

using std::cout; using std::endl;



#define tryStreaming 1
int main(int argc, char* argv[])
{

    /// Open CV stuff
    const int dst_width = 640;
    const int dst_height = 480;
    const AVRational dst_fps = {30, 1};

    // initialize OpenCV capture as input frame generator
    cv::VideoCapture cvcap(0);
    if (!cvcap.isOpened()) {
        std::cerr << "fail to open cv::VideoCapture";
        return 2;
    }
    cvcap.set(cv::CAP_PROP_FRAME_WIDTH, dst_width);
    cvcap.set(cv::CAP_PROP_FRAME_HEIGHT, dst_height);

    // allocate cv::Mat with extra bytes (required by AVFrame::data)
    std::vector<uint8_t> imgbuf(dst_height * dst_width * 3 + 16);
    cv::Mat image(dst_height, dst_width, CV_8UC3, imgbuf.data(), dst_width * 3);



    /////////////////////////
    const char *filename = "rtsp://127.0.0.1:8554/live.sdp";
    AVOutputFormat *fmt;
    AVStream *video_st;
    AVCodec *video_codec;
    double  video_time;
    int flush;
    int ret;


    //initialize FFmpeg library
    //Register all codecs and formats. */
    av_register_all();
    avformat_network_init();


    AVFormatContext *oc;
    /* allocate the output media context */
    avformat_alloc_output_context2(&oc, NULL, "rtsp", filename);


    if (!oc) {
        printf("Could not deduce output format from file extension: using MPEG.\n");
        avformat_alloc_output_context2(&oc, NULL, "mpeg", filename);
    }

    if (!oc)
        return 1;

    fmt = oc->oformat;
    if(!fmt)
        cout<<"Error creating outformat\n";



    cout<<"Codec = "<<avcodec_get_name(fmt->video_codec)<<endl;
    if (fmt->video_codec != AV_CODEC_ID_NONE)
    {

          video_st = add_stream(oc, &video_codec, fmt->video_codec);


    }

    // create new video stream
    AVCodec* vcodec = avcodec_find_encoder(fmt->video_codec);
    video_st = avformat_new_stream(oc, vcodec);
    if (!video_st) {
        std::cerr << "fail to avformat_new_stream";
        return 2;
    }
    avcodec_get_context_defaults3(video_st->codec, vcodec);
    video_st->codec->width = dst_width;
    video_st->codec->height = dst_height;
    video_st->codec->pix_fmt = vcodec->pix_fmts[0];
    video_st->codec->time_base = video_st->time_base = av_inv_q(dst_fps);
    video_st->r_frame_rate = video_st->avg_frame_rate = dst_fps;

    //video_st->codec->codec_type = AVMEDIA_TYPE_VIDEO;
    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        video_st->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    std::cout<< "format:  " << oc->oformat->name << "\n"
             << "vcodec:  " << vcodec->name << "\n"
             << "size:    " << dst_width << 'x' << dst_height << "\n"
             << "fps:     " << av_q2d(dst_fps) << "\n"
             << "pixfmt:  " << av_get_pix_fmt_name(video_st->codec->pix_fmt) << "\n"
             << std::flush;


    /////////////////////////////////
    /// AV Frame

    // initialize sample scaler
    SwsContext* swsctx = sws_getCachedContext(
        nullptr, dst_width, dst_height, AV_PIX_FMT_BGR24,
        dst_width, dst_height, video_st->codec->pix_fmt, SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!swsctx) {
        std::cerr << "fail to sws_getCachedContext";
        return 2;
    }

    // allocate frame buffer for encoding
    AVFrame* frame = av_frame_alloc();
    std::vector<uint8_t> framebuf(avpicture_get_size(video_st->codec->pix_fmt, dst_width, dst_height));
    avpicture_fill(reinterpret_cast<AVPicture*>(frame), framebuf.data(), video_st->codec->pix_fmt, dst_width, dst_height);
    frame->width = dst_width;
    frame->height = dst_height;
    frame->format = static_cast<int>(video_st->codec->pix_fmt);

    // encoding loop
    int64_t frame_pts = 0;
    unsigned nb_frames = 0;
    bool end_of_stream = false;
    int got_pkt = 0;



#if tryStreaming == 1
    av_dump_format(oc, 0, filename, 1);
    char errorBuff[80];

    if (!(fmt->flags & AVFMT_NOFILE)) {

        ret = avio_open(&oc->pb, filename, AVIO_FLAG_WRITE);
        if (ret < 0) {
            fprintf(stderr, "Could not open outfile '%s': %s",filename,av_make_error_string(errorBuff,80,ret));
            return 1;
        }
    }

    ret = avformat_write_header(oc, NULL);
    if (ret < 0){
        fprintf(stderr, "Error occurred when writing header: %s",av_make_error_string(errorBuff,80,ret));
        return 1;
    }



    /* Now that all the parameters are set, we can open the audio and
     * video codecs and allocate the necessary encode buffers. */
    if (video_st)
        open_video(oc, video_codec, video_st, frame);


#endif



    do{
        video_time = (video_st && !video_is_eof) ? video_st->pts.val * av_q2d(video_st->time_base) : INFINITY;
        if (!flush &&
            (!video_st || video_time >= STREAM_DURATION)) {
            flush = 1;
        }

        if (!end_of_stream) {
            // retrieve source image
            cvcap >> image;
            cv::imshow("press ESC to exit", image);
            if (cv::waitKey(33) == 0x1b)
                end_of_stream = true;
        }
        if (!end_of_stream) {
            // convert cv::Mat(OpenCV) to AVFrame(FFmpeg)
            const int stride[] = { static_cast<int>(image.step[0]) };
            sws_scale(swsctx, &image.data, stride, 0, image.rows, frame->data, frame->linesize);
            frame->pts = frame_pts++;
        }

#if tryStreaming == 1
        // encode video frame
        AVPacket pkt;
        pkt.data = nullptr;
        pkt.size = 0;
        av_init_packet(&pkt);
        ret = avcodec_encode_video2(video_st->codec, &pkt, end_of_stream ? nullptr : frame, &got_pkt);
        if (ret < 0) {
            std::cerr << "fail to avcodec_encode_video2: ret=" << ret << "\n";
            break;
        }



        if (got_pkt){
            // rescale packet timestamp
            pkt.duration = 1;
            av_packet_rescale_ts(&pkt, video_st->codec->time_base, video_st->time_base);
            // write packet
            //#av_write_frame(oc, &pkt);

            AVCodecContext *c = video_st->codec;
            write_frame(oc, &c->time_base, video_st, &pkt);


            std::cout << nb_frames << '\r' << std::flush;  // dump progress
            ++nb_frames;
        }
        av_free_packet(&pkt);

#else
        /* write interleaved video frames */
        if(video_st && !video_is_eof){    // && video_time < audio time
            write_video_frame(oc , video_st, flush, frame);
            //write_video_frame(oc , &pkt, flush);
        }
#endif

    } while (!end_of_stream || got_pkt);

    std::cout << nb_frames << " frames encoded" << std::endl;

    av_frame_free(&frame);
    avcodec_close(video_st->codec);

#if tryStreaming == 1

    /* Write the trailer, if any. The trailer must be written before you
     * close the CodecContexts open when you wrote the header; otherwise
     * av_write_trailer() may try to use memory that was freed on
     * av_codec_close(). */
    av_write_trailer(oc);

    /* Close each codec. */
    if (video_st)
        close_video(oc, video_st, frame);

    if (!(fmt->flags & AVFMT_NOFILE))
        /* Close the output file. */
        avio_close(oc->pb);

    /* free the stream */
    avformat_free_context(oc);
#endif

    return 0;
}

#else
int main(int argc, char** argv) {
  std::cerr << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
