from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn

import utils
import commons

import sys
import os
import re

from torch import no_grad, LongTensor
import logging

import MoeGoe

from flask import Flask, request
import flask
import time
import json
from text import japanese

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
		
		self.model_type = "unknown"
	
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


moe = MoeGoe()
modelDatas = []

currentModelIdx = -1;
def setCurrentModelIdx( num ):
	global currentModelIdx
	currentModelIdx = num
def getCurrentModelIdx():
	return currentModelIdx

exePath = os.path.dirname( sys.argv[ 0 ] )

if __name__ == '__main__':

	#
	# Initialize MoeGoe
	#

	if '--escape' in sys.argv:
		moe.escape = True
	else:
		moe.escape = False

	# delete tmp dir
	tmpdir = exePath + '/tmp'
	for f in os.listdir( tmpdir ):
		if os.path.isfile( os.path.join( tmpdir , f ) ):
			os.remove( os.path.join( tmpdir , f ) )


	#
	# WEB Server
	#
	app = Flask( __name__, static_folder = exePath, static_url_path = '' )

	@app.route( '/', methods = [ "GET" ] )
	def index():
		return app.send_static_file( 'index.html' )

	@app.route( '/', methods = [ "POST" ] )
	def command():
		
		command = request.form[ 'command' ]
		print( command )
		
		#######################################
		if command == "GetModelList":
			
			global modelDatas
			modelDatas = []
			setCurrentModelIdx( -1 )

			moe.modelpath = exePath + "/models"
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
						modeldata.model_type = "hubert"
				elif modeldata.emotion_embedding:
					modeldata.model_type = "TTS emotion"
				else:
					modeldata.model_type = "TTS"

				modelDatas.append( modeldata )
				res_data = []
				for data in modelDatas:
					res_data.append( data.toDict() )
					
			return flask.jsonify( {
				"modelDatas": res_data
			})
		
		#######################################
		elif command == "Generate":

			
			# model load
			modelIdx = int( request.form[ 'model_data_idx' ] )
			modeldata = modelDatas[ modelIdx ]
			
			# config load
			hps_ms = modeldata.hps_ms
			n_speakers = modeldata.n_speakers
			n_symbols = modeldata.n_symbols
			speakers = modeldata.speakers
			use_f0 = modeldata.use_f0
			emotion_embedding = modeldata.emotion_embedding
		
			if getCurrentModelIdx() != modelIdx:
			
				dir = exePath + "/models/" + modeldata.modelDir
				print( dir )
				model = dir + "/" + modeldata.modelFile
				print( model )

				# load
				moe.net_g_ms = SynthesizerTrn(
					n_symbols,
					hps_ms.data.filter_length // 2 + 1,
					hps_ms.train.segment_size // hps_ms.data.hop_length,
					n_speakers = n_speakers,
					emotion_embedding = emotion_embedding,
					**hps_ms.model)
				_ = moe.net_g_ms.eval()
				utils.load_checkpoint( model, moe.net_g_ms )
				
				setCurrentModelIdx( modelIdx )


			text = request.form[ 'data' ]
			speaker_id = int( request.form[ 'speaker_id' ] )
			
			isRun_marine = request.form[ 'run_marine' ]
			# run_marine
			if isRun_marine == "true":
				japanese.set_run_marine( True )
			else:
				japanese.set_run_marine( False )

			
			
			"""
			if text == '[ADVANCED]':
				text = input('Raw text:')
				print('Cleaned text is:')
				ex_print(_clean_text(
					text, hps_ms.data.text_cleaners), escape)
				continue
			"""
			
			length_scale, text = get_label_value( text, 'LENGTH', 1, 'length scale' )
			noise_scale, text = get_label_value( text, 'NOISE', 0.667, 'noise scale' )
			noise_scale_w, text = get_label_value( text, 'NOISEW', 0.8, 'deviation of noise' )

			# clener
			clener = request.form[ 'clener' ]
			isCleaned = request.form[ 'cleaned' ]
			
			cleaned, text = get_label( text, 'CLEANED' )
			
			if isCleaned == "true":
				clean_text = text
			else:
				#clean_text = _clean_text( text, hps_ms.data.text_cleaners ) if text != "" else ""
				clean_text = _clean_text( text, [ clener ] ) if text != "" else ""

			stn_tst = get_text( clean_text, hps_ms, cleaned = True )
			
			
			# generate
			try:
				with no_grad():
					x_tst = stn_tst.unsqueeze( 0 )
					x_tst_lengths = LongTensor( [ stn_tst.size( 0 ) ] )
					sid = LongTensor( [ speaker_id ] )
					audio = moe.net_g_ms.infer(
						x_tst,
						x_tst_lengths,
						sid = sid,
						noise_scale = noise_scale,
						noise_scale_w = noise_scale_w,
						length_scale = length_scale
					)[ 0 ][ 0, 0 ].data.cpu().float().numpy()
			except:
				return flask.jsonify( {
					"error": "このモデルデータは TTS 未対応か未知の形式です。"
				} )
			
			else:
				file_path = "/tmp/tmp" + str( time.time() ) + ".wav"
				out_path = exePath + file_path
				write( out_path, hps_ms.data.sampling_rate, audio )

				return flask.jsonify( {
					"audio_path": file_path,
					"clean_text": clean_text
				} )
			
		#######################################
		else:
			return flask.jsonify( {
				"error" : "Unknown command."
			} )
		
		return command


	app.run( port = 15000, debug = False )



