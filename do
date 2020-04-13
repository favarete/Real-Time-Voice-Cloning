create_voice()
{
  ffmpeg -i voice.mp3 -vn -acodec copy -ss 00:00:00 -to 00:00:10 voice_chunck.mp3
  spleeter separate -i voice_chunck.mp3 -p spleeter:2stems -o output
}