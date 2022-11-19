from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
from hubert_model import hubert_soft

import utils
import commons

import sys
import os
import re

from torch import no_grad, LongTensor
import torch

import logging

import MoeGoe

from flask import Flask, request
import flask
import time
import json
from text import japanese
import librosa
import audonnx

###########################################################################
class ModelData:
	def __init__( self ):
		self.modelDir = ""
		
		self.modelFile = ""
		self.configFile = ""
		
		self.hps_ms = None
		
		self.n_speakers = None
		self.n_symbols = None
		self.speakers = None
		self.use_f0 = None
		self.emotion_embedding = None
		
		self.text_cleaners = None
		
		self.model_type = []
	
	def toDict( self ):
		dict = {
			"modelDir": self.modelDir,
			#"modelFile": self.modelFile,
			#"configFile": self.configFile,
			"speakers": self.speakers,
			"text_cleaners": self.text_cleaners,
			"model_type": self.model_type
		}
		
		return dict


###########################################################################
class MoeGoe():

	def __init__( self ):

		self.escape = None
		self.modelpath = None
		self.net_g_ms = None

		self.hubert = None
		self.isHubert = False
		#self.W2V2 = None
		self.isW2V2 = False


##########################################################################

def get_text(text, hps, cleaned=False):
	if cleaned:
		text_norm = text_to_sequence(text, hps.symbols, [])
	else:
		text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
	if hps.data.add_blank:
		text_norm = commons.intersperse(text_norm, 0)
	text_norm = LongTensor(text_norm)
	return text_norm

def get_label_value( text, label, default, warning_name='value' ):
	value = re.search(rf'\[{label}=(.+?)\]', text)
	if value:
		try:
			text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
			value = float(value.group(1))
		except:
			print(f'Invalid {warning_name}!')
			sys.exit(1)
	else:
		value = default
	return value, text

def get_label(text, label):
	if f'[{label}]' in text:
		return True, text.replace(f'[{label}]', '')
	else:
		return False, text
##########################################################################
def getNpyDatas():
	npyDatas = []
	npyDirPath = exePath + "/models/W2V2/npy"
	tmps = os.listdir( npyDirPath )
	for f in tmps:
		if not os.path.isdir( os.path.join( npyDirPath, f ) ):
			npyDatas.append( f )
	
	return npyDatas

def setModelDatas():
	global modelDatas
	modelDatas = []
	setCurrentModelName( "" )

	moe.modelpath = exePath + "/models"
	
	# model
	files = os.listdir( moe.modelpath )
	files_dir = [ f for f in files if os.path.isdir( os.path.join( moe.modelpath, f ) ) ]
	
	for dir in files_dir:
	
		modeldata = ModelData()
		modeldata.modelDir = dir
		
		files = os.listdir( os.path.join( moe.modelpath, dir ) )
		#print( files )
		for f in files:
			if f.endswith( ".pth" ):
				modeldata.modelFile = f
				continue
			elif f.endswith( ".json" ):
				modeldata.configFile = f
				continue
		#print( modeldata.modelFile, modeldata.configFile )
		
		if modeldata.modelFile == "" or modeldata.configFile == "":
			continue
		
		modeldata.hps_ms = utils.get_hparams_from_file( os.path.join( moe.modelpath, dir + "/" + modeldata.configFile ) )
		#print( modeldata.hps_ms )
		modeldata.n_speakers = modeldata.hps_ms.data.n_speakers if 'n_speakers' in modeldata.hps_ms.data.keys() else 0
		modeldata.n_symbols = len( modeldata.hps_ms.symbols ) if 'symbols' in modeldata.hps_ms.keys() else 0
		modeldata.speakers = modeldata.hps_ms.speakers if 'speakers' in modeldata.hps_ms.keys() else [ '0' ]
		modeldata.use_f0 = modeldata.hps_ms.data.use_f0 if 'use_f0' in modeldata.hps_ms.data.keys() else False
		modeldata.emotion_embedding = modeldata.hps_ms.data.emotion_embedding if 'emotion_embedding' in modeldata.hps_ms.data.keys() else False
		modeldata.text_cleaners = modeldata.hps_ms.data.text_cleaners if 'text_cleaners' in modeldata.hps_ms.data.keys() else ""

		if modeldata.n_symbols == 0:
			if 'speakers' in modeldata.hps_ms.keys():
				modeldata.model_type.append( "HuBERT" )
		elif modeldata.emotion_embedding:
			modeldata.model_type.append( "TTS" )
			modeldata.model_type.append( "VTS" )
			modeldata.model_type.append( "W2V2" )
		else:
			modeldata.model_type.append( "TTS" )
			modeldata.model_type.append( "VTS" )

		modelDatas.append( modeldata )

	return


##########################################################################
ALLOWED_EXTENSIONS = {'txt', 'wav', 'png', 'jpg', 'jpeg', 'mp3'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
##########################################################################


moe = MoeGoe()
modelDatas = []

dev = "cpu"
if torch.cuda.is_available():
	dev = "cuda:0"

currentModelName = "";
def setCurrentModelName( name ):
	global currentModelName
	currentModelName = name
def getCurrentModelName():
	return currentModelName

exePath = os.path.dirname( sys.argv[ 0 ] )

if __name__ == '__main__':

	#
	# Initialize MoeGoe
	#

	if '--escape' in sys.argv:
		moe.escape = True
	else:
		moe.escape = False

	# check CUDA
	print( "Running on :" + dev )
	
	
	# delete tmp dir
	tmpdir = exePath + '/tmp'
	for f in os.listdir( tmpdir ):
		if os.path.isfile( os.path.join( tmpdir , f ) ):
			os.remove( os.path.join( tmpdir , f ) )

	# hubert
	hubert_dir = exePath + "/models/HuBERT"
	files = os.listdir( hubert_dir )
	for f in files:
		if f.endswith( ".pt" ):
			moe.hubert = hubert_soft( hubert_dir + "/" + f )
			#moe.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")
			moe.isHubert = "true"
			break
	
	#W2V2
	w2v2_dir = exePath + "/models/W2V2"
	files = os.listdir( w2v2_dir )
	for f in files:
		if f.endswith( ".onnx" ):
			#moe.W2V2 = audonnx.load( os.path.dirname( w2v2_dir + "/" + f ) )
			moe.isW2V2 = "true"
			break;

	#
	# WEB Server
	#
	app = Flask( __name__, static_folder = exePath, static_url_path = '' )

	#####################################################################################################################
	@app.route( '/', methods = [ "GET" ] )
	def index():
		return app.send_static_file( 'index.html' )

	#####################################################################################################################
	@app.route( '/getAllDatas', methods = [ "POST" ] )
	def getAllDatas():
	
		setModelDatas()

		res_data = []
		for data in modelDatas:
			res_data.append( data.toDict() )
		
		# npy
		npyDatas = getNpyDatas()
		
		return flask.jsonify( {
			"modelDatas": res_data,
			"isHubert": moe.isHubert,
			"npyDatas": npyDatas
		})

	#####################################################################################################################
	@app.route( '/generateAudio', methods = [ "POST" ] )
	def generateAudio():
		
		req = json.loads( request.form[ 'req' ] )
		print( req )
		
		print( "generate_type: " + req[ "generate_type" ] )
		
		# modelData
		for modelData in modelDatas:
			if modelData.modelDir == req[ "model_dirname" ] :
				break;
		# speaker_id
		speaker_id = int( req[ "speaker_id" ] )

		# config load
		hps_ms = modelData.hps_ms
		n_speakers = modelData.n_speakers
		n_symbols = modelData.n_symbols
		speakers = modelData.speakers
		use_f0 = modelData.use_f0
		emotion_embedding = modelData.emotion_embedding

		# model load
		if getCurrentModelName() != modelData.modelDir:
		
			dir = exePath + "/models/" + modelData.modelDir
			model_file_path = dir + "/" + modelData.modelFile
			print( "Loading model data: " + model_file_path )

			moe.net_g_ms = SynthesizerTrn(
				n_symbols,
				hps_ms.data.filter_length // 2 + 1,
				hps_ms.train.segment_size // hps_ms.data.hop_length,
				n_speakers = n_speakers,
				emotion_embedding = emotion_embedding,
				**hps_ms.model).to( dev )
			_ = moe.net_g_ms.eval()
			utils.load_checkpoint( model_file_path, moe.net_g_ms )
			
			setCurrentModelName( modelData.modelDir )
		
		text = ""
		file_name = ""
		
		################################################################
		if req[ "generate_type" ] == "TTS" or req[ "generate_type" ] == "W2V2":

			# run_marine
			if req[ "run_marine" ] == "true":
				japanese.set_run_marine( True )
			else:
				japanese.set_run_marine( False )
			
			# text
			cleaned = True if req[ "is_cleaned" ] == "true" else False
			text = ""
			if cleaned:
				text = req[ "text_clean" ]
			else:
				text = req[ "text_raw" ]
				clener = req[ "sel_clener" ]
				if clener == "":
					clener = modelData.text_cleaners[ 0 ]
				
				if req[ "is_auto_ja" ] == "true":
					ja_cleners = [ "zh_ja_mixture_cleaners", "cjks_cleaners", "cjke_cleaners", "cjke_cleaners2", "chinese_dialect_cleaners" ]
					if clener in ja_cleners:
						text = "[JA]" + text + "[JA]"
				
				text = _clean_text( text, [ clener ] ) if text != "" else ""

			stn_tst = get_text( text, hps_ms, cleaned = True )

			# emotion
			emotion = None
			if req[ "generate_type" ] == "W2V2":
				import numpy as np
				from torch import FloatTensor
				emotion_reference = exePath + "/models/W2V2/npy/" + req[ "emote_filename" ]
				emotion = np.load( emotion_reference )
				emotion = FloatTensor( emotion ).unsqueeze( 0 ).to( dev )
				emotion = emotion * float( req[ "emote_weight" ] )

			# generate
			try:
				with no_grad():
					x_tst = stn_tst.unsqueeze( 0 ).to( dev )
					x_tst_lengths = LongTensor( [ stn_tst.size( 0 ) ] ).to( dev )
					sid = LongTensor( [ speaker_id ] ).to( dev )
					audio = moe.net_g_ms.infer(
						x_tst,
						x_tst_lengths,
						sid = sid,
						noise_scale = float( req[ "noise" ] ),
						noise_scale_w = float( req[ "noise_w" ] ),
						length_scale = 1.0 / float( req[ "speed" ] ),
						emotion_embedding = emotion
					)[ 0 ][ 0, 0 ].data.cpu().float().numpy()
			except:
				return flask.jsonify( {
					"error": "このモデルデータは不明な形式です。"
				} )
			
			else:
				file_name = "tmp" + str( time.time() ) + ".wav"
				out_path = exePath + "/tmp/" + file_name
				print( "out_path: " + out_path )
				write( out_path, hps_ms.data.sampling_rate, audio )			

			return flask.jsonify( {
				"dist_clean_text": text,
				"dist_audio_path": file_name,
				"request": req
			} )


		################################################################
		elif req[ "generate_type" ] == "VTS":
		
			src_audio_path = exePath + "/tmp/" + req[ 'src_audio_filename' ]
			print( "src_audio_path: " + src_audio_path )
			audio = utils.load_audio_to_torch( src_audio_path, hps_ms.data.sampling_rate ).to( dev )
			y = audio.unsqueeze( 0 ).to( dev )
			spec = spectrogram_torch(
				y,
				hps_ms.data.filter_length,
				hps_ms.data.sampling_rate,
				hps_ms.data.hop_length,
				hps_ms.data.win_length,
				center = False
			).to( dev )

			spec_lengths = LongTensor( [ spec.size( -1 ) ] ).to( dev )
			sid_src = LongTensor( [ speaker_id ] ).to( dev )

			target_speaker_id = int( req[ 'target_speaker_id' ] )
			with no_grad():
				sid_tgt = LongTensor( [ target_speaker_id ] ).to( dev )
				audio = moe.net_g_ms.voice_conversion(
					spec,
					spec_lengths,
					sid_src = sid_src,
					sid_tgt = sid_tgt
				)[ 0 ][ 0, 0 ].data.cpu().float().numpy()
		
			file_name = "tmp" + str( time.time() ) + ".wav"
			out_path = exePath + "/tmp/" + file_name
			print( "out_path: " + out_path )
			write( out_path, hps_ms.data.sampling_rate, audio )
			
			
			
			return flask.jsonify( {
				"dist_clean_text": "",
				"dist_audio_path": file_name,
				"request": req
			} )

		
		################################################################
		elif req[ "generate_type" ] == "HuBERT":
			src_audio_path = exePath + "/tmp/" + req[ 'src_audio_filename' ]
			print( "src_audio_path: " + src_audio_path )

			if use_f0:
				audio, sampling_rate = librosa.load( src_audio_path, sr = hps_ms.data.sampling_rate, mono = True )
				audio16000 = librosa.resample( audio, orig_sr = sampling_rate, target_sr = 16000 )
			else:
				audio16000, sampling_rate = librosa.load( src_audio_path, sr = 16000, mono = True )


			from torch import inference_mode, FloatTensor
			import numpy as np
			with inference_mode():
				units = moe.hubert.units( FloatTensor( audio16000 ).unsqueeze( 0 ).unsqueeze( 0 ) ).squeeze( 0 ).numpy()
				if use_f0:
					f0_scale = 1.0
					f0 = librosa.pyin(
						audio,
						sr = sampling_rate,
						fmin = librosa.note_to_hz( 'C0' ),
						fmax = librosa.note_to_hz( 'C7' ),
						frame_length = 1780
					)[ 0 ]
					target_length = len( units[:, 0] )
					f0 = np.nan_to_num(
						np.interp( np.arange( 0, len( f0 ) * target_length, len( f0 ) ) / target_length,
						np.arange( 0, len( f0 ) ),
						f0
					) ) * f0_scale
					units[:, 0] = f0 / 10

			stn_tst = FloatTensor( units ).to( dev )
			with no_grad():
			    x_tst = stn_tst.unsqueeze( 0 ).to( dev )
			    x_tst_lengths = LongTensor( [ stn_tst.size( 0 ) ] ).to( dev )
			    sid = LongTensor( [ speaker_id ] ).to( dev )
			    audio = moe.net_g_ms.infer(
			    	x_tst,
			    	x_tst_lengths,
			    	sid = sid,
			    	noise_scale = float( req[ "noise" ] ),
			    	noise_scale_w = float( req[ "noise_w" ] ),
			    	length_scale = 1.0 / float( req[ "speed" ] ),
			    )[ 0 ][ 0, 0 ].data.cpu().float().numpy()

			file_name = "tmp" + str( time.time() ) + ".wav"
			out_path = exePath + "/tmp/" + file_name
			print( "out_path: " + out_path )
			write( out_path, hps_ms.data.sampling_rate, audio )

			return flask.jsonify( {
				"dist_clean_text": "",
				"dist_audio_path": file_name,
				"request": req
			} )

		################################################################
		else:
			return flask.jsonify( {
				"error": "Unknown generate_type",
				"request": req
			} )


	##############################################################################
	@app.route( '/uploadFile', methods = [ "POST" ] )
	def uploadFile():
		# check if the post request has the file part
		if 'file' not in request.files:
			print( 'uploadFile: No file part' )
			return flask.jsonify( {
				"error" : "uploadFile: No file part"
			} )
		file = request.files['file']

		# If the user does not select a file, the browser submits an
		# empty file without a filename.
		if file.filename == '':
			print( 'uploadFile: No file.filename' )
			return flask.jsonify( {
				"error" : "uploadFile: No file.filename"
			} )
			
		if file and allowed_file( file.filename ):
			from werkzeug.utils import secure_filename
			filename = secure_filename( file.filename )

			new_filename = str( time.time() ) + filename
			file_path = "/tmp/" + new_filename
			out_path = exePath + file_path

			file.save( out_path )
			
			return flask.jsonify( {
				"filepath" : new_filename
			} )
			
		return flask.jsonify( {
			"error" : "uploadFile: Unkown error"
		} )

	##############################################################################
	@app.route( '/getDummyTextDict', methods = [ "POST" ] )
	def getDummyTextDict():
		
		dicts = []
		
		dictDirs = exePath + "/dummyTextDict"
		tmps = os.listdir( dictDirs )
		dirs = [ f for f in tmps if os.path.isdir( os.path.join( dictDirs, f ) ) ]
		for dir in dirs:
			path = dictDirs + "/" + dir
			files = os.listdir( os.path.join( dictDirs, dir ) )
			for f in files:
				if f == "dict_info.json":
					with open( path + '/dict_info.json', 'r', encoding='utf-8' ) as f:
						dict_info = json.load( f )
				if f == "dict_begin_words.json":
					with open( path + '/dict_begin_words.json', 'r', encoding='utf-8' ) as f:
						dict_begin_words = json.load( f )
				if f == "dict_begin_weights.json":
					with open( path + '/dict_begin_weights.json', 'r', encoding='utf-8' ) as f:
						dict_begin_weights = json.load( f )
				if f == "dict_main.json":
					with open( path + '/dict_main.json', 'r', encoding='utf-8' ) as f:
						dict_main = json.load( f )
			dict = {
				"dict_info": dict_info,
				"dict_begin_words": dict_begin_words,
				"dict_begin_weights": dict_begin_weights,
				"dict_main": dict_main
			}
				
			dicts.append( dict )
		
		
		return flask.jsonify( {
			"dicts" : dicts
		} )

	##############################################################################
	@app.route( '/createNpy', methods = [ "GET" ] )
	def createNpyIndex():
		return app.send_static_file( 'createNpy.html' )

	##############################################################################
	@app.route( '/createNpy', methods = [ "POST" ] )
	def createNpy():
		
		if moe.isW2V2 == False :
			return flask.jsonify( {
				"filename" : audio.filename,
				"result" : "ERROR: W2V2 推論用データがありません"
			} )

		import numpy as np
		audio = request.files[ 'file' ]
		base_file_name, ext = os.path.splitext( audio.filename )
		if ext != ".wav" and ext != ".mp3":
			return flask.jsonify( {
				"filename" : audio.filename,
				"result" : "ERROR: " + ext + " : .wav or .mp3 only. "
			} )
			
		filename = "tmpAudioNpy" + ext
		audio_out_path = exePath + "/tmp/" + filename
		audio.save( audio_out_path )

		files = os.listdir( exePath + "/models/W2V2" )
		for f in files:
			if f.endswith( ".onnx" ):
				W2V2 = audonnx.load( os.path.dirname( w2v2_dir + "/" + f ) )

		audio16000, sampling_rate = librosa.load( audio_out_path, sr = 16000, mono = True )
		emotion = W2V2( audio16000, sampling_rate )[ 'hidden_states' ]
		npy_out_path = exePath + "/models/W2V2/npy/" + base_file_name
		np.save( npy_out_path, emotion.squeeze( 0 ) )

		return flask.jsonify( {
			"filename" : audio.filename,
			"result" : ".npy generate success"
		} )



	##############################################################################
	app.run( port = 15000, debug = False )



