# Source this file to use the locally extracted GStreamer 1.28 stack.
# The machine has no root access, so the .deb packages were unpacked into
# ~/gst-prefix instead of being installed system-wide.
GST_PREFIX="${GST_PREFIX:-/home/hj/gst-prefix/usr}"
export LD_LIBRARY_PATH="$GST_PREFIX/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GST_PLUGIN_PATH="$GST_PREFIX/lib/x86_64-linux-gnu/gstreamer-1.0:/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
export GST_PLUGIN_SYSTEM_PATH=""
export PYTHONPATH="$GST_PREFIX/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export GI_TYPELIB_PATH="$GST_PREFIX/lib/x86_64-linux-gnu/girepository-1.0${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"
export PATH="$GST_PREFIX/bin:$PATH"
