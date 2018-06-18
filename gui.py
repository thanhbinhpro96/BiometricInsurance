import requests
import wx
import json
import gender_predictor
import age_predictor
from flask import Flask

# API URL
url = 'http://omni-api-alsvnm-1624151644.ap-southeast-2.elb.amazonaws.com/omni-new-business-services/omni/service/biometric'

class PhotoCtrl(wx.App):
    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)
        self.frame = wx.Frame(None, title='Biometric Insurance')
        self.panel = wx.Panel(self.frame)
        self.PhotoMaxSize = 240
        wx.StaticText(self.panel, label="________________INFORMATION________________",
                      pos=(300, 25))
        wx.StaticText(self.panel, label="_____________PREDICTED PREMIUM_____________",
                      pos=(300, 175))
        instructions = 'Browse for a face image'
        img = wx.Image(240, 240)

        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY,
                                         wx.Bitmap(img))

        instructLbl = wx.StaticText(self.panel, label=instructions)
        self.photoTxt = wx.TextCtrl(self.panel, size=(200, -1))
        self.browseBtn = wx.Button(self.panel, label='Browse')
        self.resetBtn = wx.Button(self.panel, label='Reset')
        self.resetBtn.Bind(wx.EVT_BUTTON, self.onReset)
        self.browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY),
                           0, wx.ALL | wx.EXPAND, 5)
        self.mainSizer.Add(instructLbl, 0, wx.ALL, 5)
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        self.sizer.Add(self.photoTxt, 0, wx.ALL, 5)
        self.sizer.Add(self.browseBtn, 0, wx.ALL, 5)
        self.sizer.Add(self.resetBtn, 0, wx.ALL, 5)
        self.mainSizer.Add(self.sizer, 0, wx.ALL, 5)

        self.panel.SetSizer(self.mainSizer)
        self.mainSizer.Fit(self.frame)

        self.panel.Layout()
        self.frame.Show()

    def onReset(self, event):
        self.frame.Close()
        self = PhotoCtrl()
        self.frame.SetSize(600, 400)
        self.frame.SetPosition((400, 300))
        self.panel.Show()
        self.MainLoop()

    def onBrowse(self, event):
        """
        Browse for file
        """
        wildcard = "JPEG files (*.jpg)|*.jpg|(*.jpeg)|*.jpeg|(*.png)|*.png"
        dialog = wx.FileDialog(None, "Choose a file",
                               wildcard=wildcard,
                               style=wx.FC_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.photoTxt.SetValue(dialog.GetPath())
        dialog.Destroy()
        self.onView()
        self.browseBtn.Disable()


    def onView(self):
        filepath = self.photoTxt.GetValue()
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
        gender = gender_predictor.FaceCV()
        age = age_predictor.FaceCV()
        age_ = age.detect_face(filepath)
        name_, gender_ = gender.detect_face(filepath)

        info_json = export(name_, age_, gender_)
		
        headers = {
            'userName': 'Agent@csc.com',
            'profileId': '12345',
            'Content-Type': 'application/json'
        }
		
        height = 50

        for key, value in info_json.items():
            value_string = str(value).replace('_', ' ')
            wx.StaticText(self.panel, label=str(str(key).title() + " : " + value_string), pos=(300, height))
            height += 25

        response = requests.request('POST', url=url, json=info_json, headers=headers)
        result_json = json.JSONDecoder().decode(response.text)

        height += 75
        for key, value in result_json.items():
            key_string = str(key).replace('-', ' ')
            value_string = "$" + str(value)
            wx.StaticText(self.panel, label=str(key_string.title() + " : " + value_string), pos=(300, height))

        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        if W > H:
            NewW = self.PhotoMaxSize
            NewH = self.PhotoMaxSize * H / W
        else:
            NewH = self.PhotoMaxSize
            NewW = self.PhotoMaxSize * W / H
        img = img.Scale(NewW, NewH)

        self.imageCtrl.SetBitmap(wx.BitmapFromImage(img))
        self.panel.Refresh()

app = Flask(__name__)

def export(name, age, gender):
    string_result = json.JSONEncoder().encode({"name" : name, "age" : age, "gender" : gender})
    return json.JSONDecoder().decode(string_result)

def main():
    app = PhotoCtrl()
    app.frame.SetSize(600, 400)
    app.frame.SetPosition((400, 300))
    app.panel.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()