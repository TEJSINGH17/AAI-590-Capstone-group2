import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
    return True

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            print(f"Frame {frame_meta.frame_num} | "
                  f"Class: {obj_meta.obj_label} | "
                  f"Confidence: {obj_meta.confidence:.2f} | "
                  f"BBox: left={obj_meta.rect_params.left:.1f}, "
                  f"top={obj_meta.rect_params.top:.1f}, "
                  f"width={obj_meta.rect_params.width:.1f}, "
                  f"height={obj_meta.rect_params.height:.1f}")

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def main():
    input_file = "/home/logicpro09/omniview_ai/output_ds_3_reenc.mp4"

    Gst.init(None)

    pipeline = Gst.Pipeline()

    # Create elements
    source      = Gst.ElementFactory.make("filesrc", "file-source")
    decoder     = Gst.ElementFactory.make("decodebin", "decoder")
    nvvidconv0  = Gst.ElementFactory.make("nvvideoconvert", "convertor0")
    streammux   = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie        = Gst.ElementFactory.make("nvinfer", "primary-inference")
    nvvidconv1  = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    nvosd       = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvvidconv2  = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    capsfilter  = Gst.ElementFactory.make("capsfilter", "capsfilter")
    sink        = Gst.ElementFactory.make("nveglglessink", "sink")

    if not all([pipeline, source, decoder, nvvidconv0, streammux, pgie,
                nvvidconv1, nvosd, nvvidconv2, capsfilter, sink]):
        print("Failed to create elements")
        sys.exit(1)

    # Configure caps
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    capsfilter.set_property("caps", caps)

    # Configure elements
    source.set_property('location', input_file)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('nvbuf-memory-type', 0)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', '/home/logicpro09/omniview_ai/config_infer_primary.txt')
    sink.set_property('sync', False)

    # Add elements to pipeline
    for element in [source, decoder, nvvidconv0, streammux, pgie,
                    nvvidconv1, nvosd, nvvidconv2, capsfilter, sink]:
        pipeline.add(element)

    # Link static elements
    source.link(decoder)
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(sink)

    # Connect decoder to nvvidconv0 dynamically then to streammux
    def on_pad_added(decoder, pad):
        caps = pad.get_current_caps()
        if not caps:
            caps = pad.query_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        if 'video' in gstname:
            sinkpad = nvvidconv0.get_static_pad("sink")
            if pad.link(sinkpad) == Gst.PadLinkReturn.OK:
                print("Decoder linked to nvvidconv0")
                srcpad = nvvidconv0.get_static_pad("src")
                muxpad = streammux.get_request_pad("sink_0")
                if srcpad.link(muxpad) == Gst.PadLinkReturn.OK:
                    print("nvvidconv0 linked to streammux")
                else:
                    print("Failed to link nvvidconv0 to streammux")
            else:
                print("Failed to link decoder to nvvidconv0")

    decoder.connect("pad-added", on_pad_added)

    # Add probe for metadata extraction
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Start pipeline
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped")

if __name__ == '__main__':
    main()
