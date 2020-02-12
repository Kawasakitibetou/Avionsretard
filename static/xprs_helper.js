/******************************************************************************************************
 *                                             XPRS HELPER - SHARED FUNCTIONS 
 ******************************************************************************************************/

var XPRSHelper = {};
XPRSHelper.currentUser = {};


XPRSHelper.LOCAL_SERVER_PATH = "http://localhost:7000";
XPRSHelper.PRODUCTION_SERVER_PATH = "/";
XPRSHelper.RELATIVE_SERVER_PATH = "";
//XPRSHelper.devMode = "LOCAL";
XPRSHelper.devMode = "PRODUCTION";
XPRSHelper.saveQueue = {"PENDING":{},"ERRED":{},"COMPLETED":{},"FUTURE":{}};
XPRSHelper.presetTypes = {
		"APPS":{"TYPE":"PROMO","NAME":"Widgets","GROUP":"SLIDESHOWS","PAGE":1},
		"FEATURES":{"TYPE":"FEATURES","NAME":"Features","GROUP":"FEATURES","PAGE":1},
		"TEAM":{"TYPE":"TEAM","NAME":"Team","GROUP":"FEATURES","PAGE":1},
		"LOGOS":{"TYPE":"LOGOS","NAME":"Logos","GROUP":"FEATURES","PAGE":1},
		"TESTIMONIALS":{"TYPE":"TESTIMONIALS","NAME":"Testimonials","GROUP":"SLIDESHOWS","PAGE":1},
		"PROJECTS":{"TYPE":"PROJECTS","NAME":"Projects","GROUP":"FEATURES","PAGE":1},
		"GALLERIES":{"TYPE":"GALLERIES","NAME":"Gallery","GROUP":"GALLERIES","PAGE":1},
		"BLOG":{"TYPE":"BLOG","NAME":"Blog","GROUP":"GALLERIES","PAGE":1},
		"STORE":{"TYPE":"STORE","NAME":"Store","GROUP":"GALLERIES","PAGE":1},
		"TEXT_BLOCK":{"TYPE":"TEXT_BLOCK","NAME":"Text Block","GROUP":"ITEMS","PAGE":1},
		"ARTICLE":{"TYPE":"ARTICLE","NAME":"Article","GROUP":"ITEMS","PAGE":1},
		"HEADER":{"TYPE":"HEADER","NAME":"Header","GROUP":"ITEMS","PAGE":1},
		"CALL_TO_ACTION":{"TYPE":"CALL_TO_ACTION","NAME":"Call to action","GROUP":"ITEMS","PAGE":1},
		"ITEMS":{"TYPE":"ITEMS","NAME":"Item","GROUP":"ITEMS","PAGE":1},
		"PROMO":{"TYPE":"PROMO","NAME":"Header","GROUP":"SLIDESHOWS","PAGE":1},
		"FORM":{"TYPE":"FORM","NAME":"Forms","GROUP":"ITEMS","PAGE":1},
		"SLIDESHOWS":{"TYPE":"SLIDESHOWS","NAME":"Slideshow","GROUP":"SLIDESHOWS","PAGE":1},
		"FOOD_MENU":{"TYPE":"FOOD_MENU","NAME":"Food Menu","GROUP":"FEATURES","PAGE":1},
		"MAPS":{"TYPE":"MAPS","NAME":"Maps","GROUP":"SLIDESHOWS","PAGE":1},
		"VIDEOS":{"TYPE":"VIDEOS","NAME":"Videos","GROUP":"SLIDESHOWS","PAGE":1},
		"RESERVATION":{"TYPE":"RESERVATION","NAME":"Reservations","GROUP":"ITEMS","PAGE":1},
		"STORIES":{"TYPE":"STORIES","NAME":"Stories","GROUP":"FEATURES","PAGE":1},
		"PRICING":{"TYPE":"PRICING","NAME":"Pricing","GROUP":"FEATURES","PAGE":1},
		"SERVICES":{"TYPE":"SERVICES","NAME":"Services","GROUP":"FEATURES","PAGE":1},
		"SOCIAL_ICONS":{"TYPE":"SOCIAL_ICONS","NAME":"Social","GROUP":"FEATURES","PAGE":1},
		"BIO_CV":{"TYPE":"BIO_CV","NAME":"Bio / Cv","GROUP":"ITEMS","PAGE":1},
		"TABLES":{"TYPE":"TABLES","NAME":"Tables","GROUP":"FEATURES","PAGE":1},
		"MENUS":{"TYPE":"MENUS","NAME":"Menu","GROUP":"MENUS","COLOR":"#6666FF"},
		"FOOTERS":{"TYPE":"FOOTERS","NAME":"Footer","GROUP":"FOOTERS","PAGE":1},
		"SELF":{"TYPE":"SELF","NAME":"Self","GROUP":"WIDGETS","PAGE":1,"COLOR":"#5D99C2"},
		//ELEMENT
		"TITLE":{"TYPE":"TITLE","NAME":"Title","GROUP":"ELEMENTS","COLOR":"#0f95ee","PAGE":2},
		"PIC":{"TYPE":"PIC","NAME":"Pic","GROUP":"ELEMENTS","COLOR":"#00cc99","PAGE":2},
		"DRAGGABLE_PIC":{"TYPE":"DRAGGABLE_PIC","NAME":"Draggable image","GROUP":"ELEMENTS","COLOR":"#00cc99","PAGE":2},
		"SUBTITLE":{"TYPE":"SUBTITLE","NAME":"Subtitle","GROUP":"ELEMENTS","COLOR":"#336667","PAGE":2},
		"VIDEO":{"TYPE":"VIDEO","NAME":"Video","GROUP":"ELEMENTS","COLOR":"#6766cc","PAGE":2},
		"BODY":{"TYPE":"BODY","NAME":"Body","GROUP":"ELEMENTS","COLOR":"#ff679a","PAGE":2},
		"QUOTE":{"TYPE":"QUOTE","NAME":"Quote","GROUP":"ELEMENTS","COLOR":"#FF9933","PAGE":2},
		"LINK":{"TYPE":"LINK","NAME":"Link","GROUP":"ELEMENTS","COLOR":"#663398","PAGE":2},
		"ICON":{"TYPE":"ICON","NAME":"Icon","GROUP":"ELEMENTS","COLOR":"#996533","PAGE":2},
		"MAP":{"TYPE":"MAP","NAME":"Map","GROUP":"ELEMENTS","COLOR":"#0099cb","PAGE":2},
		"HTML":{"TYPE":"HTML","NAME":"HTML","GROUP":"UNRESOLVED","COLOR":"#999999","PAGE":0},
		"DIVIDER":{"TYPE":"DIVIDER","NAME":"Divider","GROUP":"UNRESOLVED","COLOR":"#BDB76B","PAGE":0},
		"RAW":{"TYPE":"RAW","NAME":"Raw","GROUP":"ELEMENTS","COLOR":"#5a5a5a","PAGE":2},
		//UNRESOLVED
		"UNRESOLVED":{"TYPE":"UNRESOLVED","NAME":"","GROUP":"UNRESOLVED","COLOR":"","PAGE":0},
		"SOCIAL":{"TYPE":"SOCIAL","NAME":"","GROUP":"UNRESOLVED","COLOR":"#333","PAGE":0},
		"IMAGE":{"TYPE":"IMAGE","NAME":"Pic","GROUP":"UNRESOLVED","COLOR":"#00cc99","PAGE":0},
		"LABEL":{"TYPE":"LABEL","NAME":"Label","GROUP":"UNRESOLVED","COLOR":"#663398","PAGE":0},
		"FIELD":{"TYPE":"FIELD","NAME":"Field","GROUP":"UNRESOLVED","COLOR":"#663398","PAGE":0},
		"PRICE":{"TYPE":"PRICE","NAME":"PRICE","GROUP":"UNRESOLVED","COLOR":"#339966","PAGE":0}, // ECOMMERCE
		"CART":{"TYPE":"CART","NAME":"Cart","GROUP":"UNRESOLVED","COLOR":"#00CC99","PAGE":0}, // ECOMMERCE
		"QUOTE_AUTHOR":{"TYPE":"QUOTE_AUTHOR","NAME":"Quote Author","GROUP":"UNRESOLVED","COLOR":"#FF9933","PAGE":0},
		"INLINE_PIC":{"TYPE":"INLINE_PIC","NAME":"Image","GROUP":"UNRESOLVED","COLOR":"#00CC99","PAGE":0}
};





XPRSHelper.inPresetGroup = function(presetId,presetGroup){
	if (presetId in XPRSHelper.presetTypes){
		return  (XPRSHelper.presetTypes[presetId]["GROUP"] == presetGroup);
	}
	return false;
};

XPRSHelper.isManagable = function(presetId){
	if (presetId in XPRSHelper.presetTypes){
		return  (XPRSHelper.presetTypes[presetId]["GROUP"] in {"FEATURES":true,"SLIDESHOWS":true,"GALLERIES":true});
	}
	return false;
};

XPRSHelper.getServerPath = function(){
	if ($("body").attr("data-server")){
		return $("body").attr("data-server");
	}else{
		return XPRSHelper.RELATIVE_SERVER_PATH;	
	}
	
	
//	if (XPRSHelper.devMode=="LOCAL"){
//		return XPRSHelper.LOCAL_SERVER_PATH;
//	}else{
//		return XPRSHelper.PRODUCTION_SERVER_PATH
//	}
};


XPRSHelper.getStaticServerPath = function(){
	return $("body").attr("data-static-server");
};


XPRSHelper.getXprsCookie = function(cookieName){
	cookieName = cookieName.replace("xprs","imxprs");
	try {
		return $.cookie(cookieName);
	} catch(err) {
		var name = cookieName + "=";
	    var ca = document.cookie.split(';');
	    for(var i = 0; i <ca.length; i++) {
	        var c = ca[i];
	        while (c.charAt(0)==' ') {
	            c = c.substring(1);
	        }
	        if (c.indexOf(name) == 0) {
	            return c.substring(name.length,c.length);
	        }
	    }
	}
	
};

XPRSHelper.setXprsCookie = function(cookieName,cookieValue){
	cookieName = cookieName.replace("xprs","imxprs");
	if (window.location.href.indexOf("imcreator.com") == -1){
		$.cookie(cookieName,cookieValue, { path: '/' });
	}else{
		document.cookie = cookieName + '=' + cookieValue + '; expires=Fri, 27 Jul 2019 02:47:11 UTC; domain=.imcreator.com; path=/';
	}
};

XPRSHelper.removeXprsCookie = function(cookieName){
	cookieName = cookieName.replace("xprs","imxprs");
	if (window.location.href.indexOf("imcreator.com") == -1){
		$.removeCookie(cookieName, { path: '/' });
	}else{
		document.cookie = cookieName + '=invalid; expires=Thu, 01 Jan 1970 00:00:00 GMT; domain=.imcreator.com; path=/';
	}
};

XPRSHelper.getShoprocketUrl = function(){
	var dashboardUrl = $("body").attr("data-ecommerce-dashboard");
	if (dashboardUrl){
		return dashboardUrl;
	}else{
		return "https://dashboard.shoprocket.co";
	}
};

XPRSHelper.getUrlParameterByName = function(name) {
    name = name.replace(/[\[]/, "\\[").replace(/[\]]/, "\\]");
    var regex = new RegExp("[\\?&]" + name + "=([^&#]*)"),
        results = regex.exec(location.search);
    return results == null ? "" : decodeURIComponent(results[1].replace(/\+/g, " "));
};

XPRSHelper.updateParent = function(msg) {
	XPRSHelper.getParentWindow().postMessage(msg, '*');
};


XPRSHelper.GET = function(getPath, params, callbackFunc,responseType){
	if (getPath == "SKIP"){
		if (typeof callbackFunc != "undefined"){
			callbackFunc({});
		}
		return;
	}
	if (typeof EditorHelper != "undefined"){
		params["root_id"] = EditorHelper.getRootId();
	}else if (typeof SpimeDualView != "undefined"){
		params["root_id"] = SpimeDualView.getRootId();
	}
	$.get(XPRSHelper.getServerPath() + getPath, params, function(data) {
		if (typeof callbackFunc != "undefined"){
			callbackFunc(data);
		}
	},responseType).fail(function(xhr, textStatus, errorThrown) {
		try{
			XPRSHelper.reportError("something went wrong... " + textStatus  + " " + errorThrown,{"ajax_url":getPath,"ajax_params":JSON.stringify(params)});
		    window.console.error(text);
		}catch (ex) {}
	 });
};

XPRSHelper.GETCORS = function(postPath, params, callbackFunc,responseType){
	$.ajax({
		  type: "GET",
		  url: XPRSHelper.getServerPath() + postPath,
		  data: params,
		  xhrFields: {
	           withCredentials: true
	      },
	      crossDomain: true,
		  success: function(data) {
				if (typeof callbackFunc != "undefined"){
					callbackFunc(data);
				}
		  },
		  dataType: responseType
		}).fail(function(xhr, textStatus, errorThrown) {
			try{
				XPRSHelper.reportError("something went wrong... " + textStatus  + " " + errorThrown,{"ajax_url":getPath,"ajax_params":JSON.stringify(params)});
			    window.console.error(text);
			}catch (ex) {}
		 });
};


XPRSHelper.POST = function(postPath, params, callbackFunc,responseType,callbackOnly){
	if (callbackOnly){
		if (typeof callbackFunc != "undefined"){
			callbackFunc();
		} 
		return;
	}
	if (typeof EditorHelper != "undefined"){
		params["root_id"] = EditorHelper.getRootId();
	}else if (typeof SpimeDualView != "undefined"){
		params["root_id"] = SpimeDualView.getRootId();
	}
	$.ajax({
		  type: "POST",
		  url: XPRSHelper.getServerPath() + postPath,
		  data: params,
		  success: function(data) {
				if (typeof callbackFunc != "undefined"){
					callbackFunc(data);
				}
		  },
		  dataType: responseType
		}).fail(function(xhr, textStatus, errorThrown) {
			try{
				XPRSHelper.reportError("something went wrong... " + textStatus  + " " + errorThrown,{"ajax_url":getPath,"ajax_params":JSON.stringify(params)});
			    window.console.error(text);
			}catch (ex) {}
		 });
};

XPRSHelper.POSTCORS = function(postPath, params, callbackFunc,responseType){
	$.ajax({
		  type: "POST",
		  xhrFields: {
	           withCredentials: true
	      },
	      crossDomain: true,
		  url: XPRSHelper.getServerPath() + postPath,
		  data: params,
		  success: function(data) {
				if (typeof callbackFunc != "undefined"){
					callbackFunc(data);
				}
		  },
		  dataType: responseType
		}).fail(function(xhr, textStatus, errorThrown) {
			try{
				XPRSHelper.reportError("something went wrong... " + textStatus  + " " + errorThrown,{"ajax_url":getPath,"ajax_params":JSON.stringify(params)});
			    window.console.error(text);
			}catch (ex) {}
		 });
};


XPRSHelper.SAFEPOST = function(url,params,saveKey,saveName,callbackFunc,callbackOnly){
	if (callbackOnly){
		if (typeof callbackFunc != "undefined"){
			callbackFunc();
		} 
		return;
	}
	if (XPRSHelper.pendingActionExists(saveKey) || XPRSHelper.futureQueueSize(saveKey) > 0){
		if (XPRSHelper.pendingActionExists(saveKey)){
		}else if (XPRSHelper.futureQueueSize(saveKey) > 0){
		}
		XPRSHelper.addToFutureQueue(saveKey,url,saveName,params,callbackFunc);
		return;
	}else{
		XPRSHelper.markAsPending(saveKey);
	}
	if (typeof EditorHelper != "undefined"){
		params["root_id"] = EditorHelper.getRootId();
	}else if (typeof SpimeDualView != "undefined"){
		params["root_id"] = SpimeDualView.getRootId();
	}
	$.ajax({
		  type: "POST",
		  url: XPRSHelper.getServerPath() + url,
		  data: params,
		  success: function(result) {
			  if (result.response == "SUCCESS"){
				  XPRSHelper.updateSaveQueue(saveKey,"PENDING","COMPLETED",result);
				  XPRSHelper.updateParent({"deliver_to":"parent","action":"saved"});
				  if (typeof callbackFunc != "undefined"){
						callbackFunc(result);
				  }
				  if (typeof XPRSUndo != "undefined"){
					  XPRSUndo.pushHistoryEntry({"key":saveKey,"url":url,"name":saveName,"params":params});
				  }
			  }else{
				  //handle error!!
				  console.log("got error for key " + saveKey + " with result " + JSON.stringify(result));
				  XPRSHelper.updateSaveQueue(saveKey,"PENDING","ERRED",result);
			  }
		  },
		  dataType: "json"
		}).fail(function(xhr, textStatus, errorThrown) {
			try{
				XPRSHelper.reportError("something went wrong... " + textStatus  + " " + errorThrown,{"ajax_url":url,"ajax_params":JSON.stringify(params)});
			    window.console.error(text);
			}catch (ex) {}
		 }).always(function(){
			 if (XPRSHelper.futureQueueSize(saveKey) > 0){
				var nextAction = XPRSHelper.getNextSaveAction(saveKey);
				if (nextAction != null){
					console.log("found a new action, for key "  + saveKey + " calling future action")
					setTimeout(function(){
						XPRSHelper.SAFEPOST(nextAction.url,nextAction.params,nextAction.saveKey,nextAction.saveName,nextAction.callback);	
					},1000)
					
				}
			}
		 });
	
};


XPRSHelper.pendingActionExists = function(saveKey){
	return (saveKey in XPRSHelper.saveQueue["PENDING"]);
};

XPRSHelper.markAsPending = function(saveKey){
	XPRSHelper.saveQueue["PENDING"][saveKey] = true;
};



XPRSHelper.addToFutureQueue = function(saveKey,url,saveName,params,callback){
	var action = {};
	action.url = url;
	action.params = params;
	action.saveKey = saveKey;
	action.saveName = saveName;
	action.callback = callback;
	if (!(saveKey in XPRSHelper.saveQueue["FUTURE"])){
		XPRSHelper.saveQueue["FUTURE"][saveKey] = [];
	}
	XPRSHelper.saveQueue["FUTURE"][saveKey].push(action);
};

XPRSHelper.futureQueueSize = function(saveKey){
	var size = 0;
	if (saveKey in XPRSHelper.saveQueue["FUTURE"]){
		size = XPRSHelper.saveQueue["FUTURE"][saveKey].length
	}
	return size;
};

XPRSHelper.updateSaveQueue = function(saveKey,fromState,toState,result){
	XPRSHelper.saveQueue[toState][saveKey] = true;
	delete XPRSHelper.saveQueue[fromState][saveKey];
};

XPRSHelper.getNextSaveAction = function(saveKey){
	var nextAction = null;
	if (saveKey in XPRSHelper.saveQueue["FUTURE"]){
		var nextAction = XPRSHelper.saveQueue["FUTURE"][saveKey].pop();
		if (XPRSHelper.saveQueue["FUTURE"][saveKey].length == 0){
			delete XPRSHelper.saveQueue["FUTURE"][saveKey];
		}
	}
	return nextAction;
};





XPRSHelper.localServer = function(){
	try{
		return (XPRSHelper.getParentWindow().location.href.indexOf("localhost") != -1);
	}catch(err){};
};


XPRSHelper.clonePrefix = function() {
	return 'xxxxx'.replace(/[xy]/g, function(c) {
		var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
		return v.toString(16);
	});
};

XPRSHelper.onCssTransitionFinish = function(obj,callbackFunc){
	//obj.one('transitionend webkitTransitionEnd oTransitionEnd otransitionend MSTransitionEnd',function(event) {
		//event.stopPropagation();
		//$(this).unbind('transitionend webkitTransitionEnd oTransitionEnd otransitionend MSTransitionEnd');
		if (typeof callbackFunc != "undefined"){
			callbackFunc();
		}
	//});
};

XPRSHelper.signout = function(){
	XPRSHelper.removeXprsCookie("xprs_user");
	XPRSHelper.removeXprsCookie("xprs_root");
	XPRSHelper.removeXprsCookie("xprs_session");
	XPRSHelper.removeXprsCookie("xprs_email");
//	$.removeCookie('xprs_user', { path: '/' });
//	$.removeCookie('xprs_root', { path: '/' });
//	$.removeCookie('xprs_session', { path: '/' });
//	document.cookie = "xprs_user=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
//	document.cookie = "xprs_root=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
//	document.cookie = "xprs_session=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
//	document.cookie = "xprs_user=;expires=Thu, 01 Jan 1970 00:00:00 GMT;domain=.imcreator.com; path=/";
//	document.cookie = "xprs_root=;expires=Thu, 01 Jan 1970 00:00:00 GMT;domain=.imcreator.com; path=/";
//	document.cookie = "xprs_session=;expires=Thu, 01 Jan 1970 00:00:00 GMT;domain=.imcreator.com; path=/";
	XPRSHelper.getParentWindow().location.href = "/";
};


XPRSHelper.getCurrentUser = function(callbackFunc){
	XPRSHelper.GET("/get_loggedin_user",{},function(userObj){
		XPRSHelper.currentUser = userObj;
		if (typeof callbackFunc != "undefined"){
			callbackFunc();
		}
	},"json");
//	
//	var userFromCookie = decodeURIComponent(XPRSHelper.getXprsCookie("xprs_user"));
//	var sessionFromCookie = XPRSHelper.getXprsCookie("xprs_session");
//	if (typeof userFromCookie == "undefined" || typeof sessionFromCookie == "undefined"){
//		return "guest";
//	}else{
//		if (sessionFromCookie != "invalid"){
//			return userFromCookie;
//		}
//	}
//	return "guest";
};

XPRSHelper.getCurrentDate = function() {
	return dateFormat(new Date(), 'UTC:dd-mm-yyyy HH:MM:ss');
};

XPRSHelper.trackEvent = function(eventName,category,label,skipBi){
	if (typeof analytics != "undefined"){
		analytics.track(eventName,{label:label});
		analytics.page(eventName,{title:eventName, url:"/" + eventName, path: "/" + eventName});
		
		if (typeof ga != "undefined"){
			if (typeof ANALYTICS_CODES != "undefined"){
				for (i in ANALYTICS_CODES){
			  		var analyticsName = ANALYTICS_CODES[i]["name"];
			  		ga(analyticsName + '.send', {
						  'hitType': 'event',          // Required.
						  'eventCategory': category,   // Required.
						  'eventAction': eventName,      // Required.
						  'eventLabel': label
					});
					ga(analyticsName + '.send', 'pageview', {
						  'page': eventName,
						  'title':eventName
					});
				}
			}
			
		}
		try{
			if (typeof IMOS != "undefined"){
				var ourGoals = {"Registration":true,"Premium":true,"Whitelabel Premium":true};
				if (eventName in ourGoals){
					//will be tracked from the goal function
					//IMOS.trackGoal(eventName);
				}else{
					IMOS.trackEvent(eventName);	
				}
			}
		} catch (err){
			console.log("failed to track imos");
		}
	}else{
		XPRSHelper.updateParent({"deliver_to":"parent","action":"track-event","event_name":eventName,"category":category,"label":label});
		return;
	}
	if (!skipBi){
		XPRSHelper.POST("/track_user_action", {"event_name":eventName,"url":window.location.href,"extra_data":label});
	}
};

XPRSHelper.identifyTemplate = function(templateName){
	if (typeof analytics != "undefined"){
		analytics.identify({templates: templateName});
	}
};

XPRSHelper.getParentWindow = function(){
	if (typeof SpimeDualView != "undefined"){
		return window.self
	}
	try { 
		if (window.parent.location.href){
			return window.parent;
		}
	}catch (err){
		return window.self;
	}
};


XPRSHelper.changeHash = function(newHash){
	XPRSHelper.getParentWindow().location.hash = newHash;
};

XPRSHelper.getHash = function(){
	return XPRSHelper.getParentWindow().location.hash;
};


XPRSHelper.imagePreloader = function(arrayOfImages,containingFolder,suffix) {
    $(arrayOfImages).each(function(){
        $('<img/>')[0].src = containingFolder + "/" + this + "." + suffix;
    });
};



XPRSHelper.renderTip = function(tipIndex){
	tooltTips = [
		     		{
			     		"category":"Adding Sections",
			     		"title":"Add a new section to your page",
			     		"content":"Click to add  text, pictures, gallery, slideshow, contact form and more.",
			     		"direction":"bottom-left",
			     		"selector":"#add-stripe",
			     		"container_selector":".master.item-box:visible:not(.force-transparency):not(.element-box):not(.header-box) + .control-handle:eq(0) .add-stripe-holder",
			     		"frame":"viewer",
			     		"circle_offset_left":38,
			     		"circle_offset_top":40//,
			     		//"generate_click":true
			     		
			     	},
		     		{
			     		"category":"Editing Content",
			     		"title":"Click & edit any element",
			     		"content":"Click any element and insert your own content: text, pictures and more.",
			     		"direction":"bottom-right",
			     		"selector":".text-element",
			     		"container_selector":".master.item-box:not(.header-box)",
			     		"frame":"viewer",
			     		"circle_offset_left":0,
			     		"circle_offset_top":37
			     	},
		     		{
			     		"category":"Adding Pages",
			     		"title":"Add a new page to your site",
			     		"content":"Click to add a new page: about, blog, gallery, contact and more",
			     		"direction":"top-right",
			     		"selector":"#pages-menu-btn",
			     		"container_selector":"#control-panel-left",
			     		"frame":"dual",
			     		"circle_offset_left":15,
			     		"circle_offset_top":29
		     		},
		     		{
			     		"category":"Responsive",
			     		"title":"Preview on all devices",
			     		"content":"See how your website looks on tablets and mobile phones",
			     		"direction":"top-right",
			     		"selector":"#preview-menu-btn",
			     		"container_selector":"#control-panel-right",
			     		"frame":"dual",
			     		"circle_offset_left":8,
			     		"circle_offset_top":27
			     	},
		     		{
			     		"category":"Publish",
			     		"title":"Publish your site",
			     		"content":"When you finish editing your site, click Publish to connect to a domain and share your site with the world.",
			     		"direction":"top-right",
			     		"selector":"#publish-btn",
			     		"container_selector":"body",
			     		"frame":"dual",
			     		"circle_offset_left":-19,
			     		"circle_offset_top":28
			     	}
		     	];
	tipObj = tooltTips[tipIndex];
	if (tipObj.frame == "dual"){
		if (typeof SpimeDualView == "undefined"){
			XPRSHelper.updateParent({"deliver_to":"parent","action":"show-tooltip","tip":tipIndex});
			return;
		}
	}
	if (tipObj.frame == "viewer"){
		if (typeof EditorHelper == "undefined"){
			XPRSHelper.updateParent({"deliver_to":"viewer","action":"show-tooltip","tip":tipIndex});
			return;
		}
	}
	var tooltipWrapper = $("<div class='tooltip-wrapper tooltip-ui tooltip-ui"+ tipIndex +"'  />");
	var tooltipHolder = $("<div class='tooltip-holder' />");
	var availableContainers = $(tipObj.container_selector);
	var tooltipRefrenceContainer = availableContainers.first();
	
	
	

	
	var tooltipRefrenceElement = tooltipRefrenceContainer.find(tipObj.selector).first();
	if (tooltipRefrenceElement.length == 0){
		if (availableContainers.length > 1){
			tooltipRefrenceContainer = availableContainers.eq(1);
			tooltipRefrenceElement = tooltipRefrenceContainer.find(tipObj.selector);
		}
	}
	var lastTip = false;
	var nextTipIndex = tipIndex + 1;
	if (nextTipIndex == tooltTips.length){
		lastTip = true;
	}
	//console.log(tooltipRefrenceContainer.attr("class") + " ---- " + tooltipRefrenceElement.attr("class"))
	//No such reference element, skip
	if (tooltipRefrenceElement.length == 0){
		XPRSHelper.renderTip(nextTipIndex);
		console.log("didn't find")
		return;
	}
	
	
	
	
	var tooltipcategory = $("<div class='tooltip-category t-t' />").text(tipObj.category);
	var tooltipTitle = $("<div class='tooltip-title t-t' />").text(tipObj.title);
	var tooltipContent = $("<div class='tooltip-content t-t' />").html(tipObj.content);
	
	nextBtnText = (lastTip) ? "got it" : "next";
	
	var tooltipNextTip = $("<div class='tooltip-next tooltip-btn t-t' />").text(nextBtnText);
	var tooltipHideTip = $("<div class='tooltip-hide tooltip-btn t-t' />").text("hide");
	
	tooltipNextTip.unbind("click").bind("click",function(e){
		e.stopPropagation();
		$(".tooltip-ui" + tipIndex).remove();
		if (tipObj.generate_click){
			tooltipRefrenceElement.trigger("click");
		}
		XPRSHelper.updateParent({"deliver_to":"viewer","action":"remove-tooltips", "tooltip_index":tipIndex});
		if ( !lastTip ){
			XPRSHelper.renderTip(nextTipIndex);
		}else{
			XPRSHelper.updateParent({"deliver_to":"parent","action":"remove-tooltips", "tooltip_index":tipIndex});
		}
	});
	
	var tooltipCircle = $("<div class='tooltip-circle tooltip-ui tooltip-ui"+ tipIndex +"'' />");
	
	tooltipCircle.unbind("click").bind("click",function(event) {
		event.stopPropagation();
		XPRSHelper.updateParent({"deliver_to":"parent","action":"remove-tooltips"});
		tooltipRefrenceElement.click();
	});
	
	if (tipObj.generate_click){
		setTimeout(function(){
			tooltipRefrenceElement.trigger("click");
		},1500)
	}
	
	tooltipHideTip.unbind("click").bind("click",function(e){
		e.stopPropagation();
		XPRSHelper.updateParent({"deliver_to":"viewer","action":"remove-tooltips", "tooltip_index":tipIndex});
		XPRSHelper.updateParent({"deliver_to":"parent","action":"remove-tooltips", "tooltip_index":tipIndex});
	});
	
	
	
	tooltipHolder.append(tooltipcategory).append(tooltipTitle).append(tooltipContent).append(tooltipHideTip).append(tooltipNextTip);
	tooltipWrapper.append(tooltipHolder);
	
	tooltipRefrenceContainer.append(tooltipCircle);
	tooltipRefrenceContainer.append(tooltipWrapper);

	
	var scrollOffset = 0;
	if (tipObj.frame == "viewer"){
		scrollOffset = $('.main-page').scrollTop();
	}
	
	//console.log("positioning circle... reference element left is " + tooltipRefrenceElement.offset().left + " container left " + tooltipRefrenceContainer.offset().left)
	//console.log("positioning circle... reference element offset top is " + tooltipRefrenceElement.offset().top  + " reference element position top is " + tooltipRefrenceElement.position().top  +" container offset top " + tooltipRefrenceContainer.offset().top + " container position " + tooltipRefrenceContainer.position().top + " scrollOffset is " + scrollOffset)
	
	//tooltipCircle.css("left",tooltipRefrenceElement.offset().left - tooltipRefrenceContainer.offset().left - tipObj.circle_offset_left);
	//tooltipCircle.css("top",parseInt(tooltipRefrenceElement.offset().top) - tooltipRefrenceContainer.offset().top - tipObj.circle_offset_top + scrollOffset);
	
	tooltipCircle.offset({ top: tooltipRefrenceElement.offset().top - tipObj.circle_offset_top, left: tooltipRefrenceElement.offset().left - tipObj.circle_offset_left});
	
	
	//testing if bottom or top
	//trying to place the tip in the bottom
	//var calculatedTop = tooltipCircle.position().top - tooltipWrapper.innerHeight() + 40;
	//if (calculatedTop < documentTop){
	
	var preferredDirection = tipObj.direction;
	var calculatedLeft = 0;
	var calculatedTop = 0;
	if (preferredDirection.indexOf("bottom") != -1){
		calculatedTop = tooltipCircle.position().top - tooltipWrapper.innerHeight() + 40;
	//	console.log("we prefer bottom " + calculatedTop)
	}else{
		calculatedTop = tooltipCircle.position().top + tooltipCircle.innerHeight() - 40;
	}
	
	if (preferredDirection.indexOf("left") != -1){
		calculatedLeft = tooltipCircle.position().left + tooltipCircle.innerWidth() - 40;
	}else{
		calculatedLeft = tooltipCircle.position().left - tooltipWrapper.innerHeight();
	}
	
	if (calculatedTop < 0 && preferredDirection.indexOf("bottom") != -1 && tipObj.selector != "#add-stripe"){
		//console.log("we prefer bottom but there is no place " + calculatedTop + " < 0 " )
		preferredDirection = preferredDirection.replace("bottom","top");
		calculatedTop = tooltipCircle.position().top + tooltipCircle.innerHeight() - 40;
	}
	
	if (calculatedLeft < 0 && preferredDirection.indexOf("right") != -1 ){
		preferredDirection = preferredDirection.replace("right","left");
		calculatedLeft = tooltipCircle.position().left + tooltipCircle.innerWidth() - 40;
	}else if(calculatedLeft +  tooltipWrapper.width() > parseInt($("document").width()) && preferredDirection.indexOf("left") != -1){
		preferredDirection = preferredDirection.replace("right","left");
		calculatedLeft = tooltipCircle.position().left - tooltipWrapper.innerHeight();
	}
	
	tooltipWrapper.css("left",calculatedLeft);
	tooltipWrapper.css("top",calculatedTop );
	tooltipWrapper.addClass(preferredDirection);
	
	
	if (tipObj.frame == "viewer"){
		var topmostComponent = Math.min(tooltipCircle.offset().top,tooltipWrapper.offset().top);
		var scrollto = topmostComponent - $('.main-page').offset().top + $('.main-page').scrollTop();
		
	    var offset = tooltipCircle.offset().top;
	    var visibleAreaStart = $(window).scrollTop();
	    var visibleAreaEnd = visibleAreaStart + window.innerHeight;
	    if(offset < visibleAreaStart || offset > visibleAreaEnd){
	         // Not in view so scroll to it
	    	$('body,html').animate({scrollTop:scrollto},2000,'easeOutQuart');
	    }
	}
	
	XPRSTranslator.translateDom(tooltipWrapper);
	
	setTimeout(function(){
		tooltipRefrenceContainer.addClass("tip-highlight");
	},500);

};


XPRSHelper.xprsAlert = function(msg,params){
	if (typeof swal == "undefined"){
		console.error("XPRS Error: " + msg + " " + JSON.stringify(params));
		return;
	}
	
	if (typeof params == "undefined"){
		params = {};
		params["title"] = "_";
	}
	params["confirmButtonColor"]="#0099CC";
	params["customClass"] = "xprs-alert";
	if (params && !params["do_not_translate"]){
		msg = XPRSTranslator.translateText(msg);
	}
	params["title"] = XPRSTranslator.translateText(params["title"]);
	params["confirmButtonText"] = XPRSTranslator.translateText(params["confirmButtonText"]);
	params["cancelButtonText"] = XPRSTranslator.translateText(params["cancelButtonText"]);
	params["text"] = msg;
	
	var existingAlert = ($(".sweet-alert.visible").length == 1);
	
	if (existingAlert){
		//params["closeOnConfirm"] = false;
		//params["closeOnCancel"] = false;
		
	}

	swal(params,params["callbackfunc"]);
	if (typeof params.report_error != "undefined"){
		XPRSHelper.reportError(msg,params);
	}
	
	$(".sweet-overlay").unbind("click").bind("click",function(e){
		e.stopPropagation();
		swal.close();
	});
};


XPRSHelper.reportError = function(errorMsg,params){
	try{
		$.ajax({
			  type: "POST",
			  url: XPRSHelper.getServerPath() + "/log",
			  data: {"log_info":errorMsg,"stack_trace":Error().stack,"url":window.location.href,"ajax_url":params["ajax_url"],"ajax_params":params["ajax_params"]}
		});
	}catch (ex) {}
};


XPRSHelper.invokeLogin = function(callbackFunc,form,cancelCallback){
	//var currentUser = XPRSHelper.getCurrentUser();
	XPRSHelper.getCurrentUser(function(){
		var nextUrl = XPRSHelper.getUrlParameterByName("requested_url");
		var forceDialog = false;
		if (form && form.indexOf("force-") != -1){
			form = form.replace("force-","");
			forceDialog = true;
		}
		if (XPRSHelper.currentUser["type"] == "GUEST" || nextUrl != "" || forceDialog){
			if (typeof YSBApp != "undefined"){
				YSBApp.send({"action": "session-expired", "appid": "ywebsite"});
				return;
			}
			if (typeof ExternalLogin != "undefined"){
				if (typeof SpimeDualView != "undefined"){
					SpimeDualView.handleNavigationConfirmation = false;
				}
				if (form == "login"){
					window.location.href = ExternalLogin.loginUrl;
				}else{
					window.location.href = ExternalLogin.registerUrl;
				}
				return
			}
			if (typeof LoginModule == "undefined"){
			 var login_css = $("<link>");
			 login_css.attr({ 
			      rel:  "stylesheet",
			      type: "text/css",
			      href: XPRSHelper.getServerPath() + "/css/login.css?v=141"
			    });
			 $("head").append(login_css);
			 
			 
			 $.ajax({
				  url: XPRSHelper.getServerPath() + "/js/login.js?v=141",
				  dataType: 'script',
				  success: function(){
					  setTimeout(function(){
						  LoginModule.popLogin(callbackFunc,form,cancelCallback);
					  },250); 
				  },
				  cache: true
				  //async: false
				});
		}else{
				LoginModule.popLogin(callbackFunc,form,cancelCallback);
			}
		}else{
			if (typeof callbackFunc != "undefined"){
				callbackFunc();
			}
		}
	});
};

XPRSHelper.md5 = function(str){
		  //  discuss at: http://phpjs.org/functions/md5/
		  // original by: Webtoolkit.info (http://www.webtoolkit.info/)
		  // improved by: Michael White (http://getsprink.com)
		  // improved by: Jack
		  // improved by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
		  //    input by: Brett Zamir (http://brett-zamir.me)
		  // bugfixed by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
		  //  depends on: utf8_encode
		  //   example 1: md5('Kevin van Zonneveld');
		  //   returns 1: '6e658d4bfcb59cc13f96c14450ac40b9'

		  var xl;

		  var rotateLeft = function(lValue, iShiftBits) {
		    return (lValue << iShiftBits) | (lValue >>> (32 - iShiftBits));
		  };

		  var addUnsigned = function(lX, lY) {
		    var lX4, lY4, lX8, lY8, lResult;
		    lX8 = (lX & 0x80000000);
		    lY8 = (lY & 0x80000000);
		    lX4 = (lX & 0x40000000);
		    lY4 = (lY & 0x40000000);
		    lResult = (lX & 0x3FFFFFFF) + (lY & 0x3FFFFFFF);
		    if (lX4 & lY4) {
		      return (lResult ^ 0x80000000 ^ lX8 ^ lY8);
		    }
		    if (lX4 | lY4) {
		      if (lResult & 0x40000000) {
		        return (lResult ^ 0xC0000000 ^ lX8 ^ lY8);
		      } else {
		        return (lResult ^ 0x40000000 ^ lX8 ^ lY8);
		      }
		    } else {
		      return (lResult ^ lX8 ^ lY8);
		    }
		  };

		  var _F = function(x, y, z) {
		    return (x & y) | ((~x) & z);
		  };
		  var _G = function(x, y, z) {
		    return (x & z) | (y & (~z));
		  };
		  var _H = function(x, y, z) {
		    return (x ^ y ^ z);
		  };
		  var _I = function(x, y, z) {
		    return (y ^ (x | (~z)));
		  };

		  var _FF = function(a, b, c, d, x, s, ac) {
		    a = addUnsigned(a, addUnsigned(addUnsigned(_F(b, c, d), x), ac));
		    return addUnsigned(rotateLeft(a, s), b);
		  };

		  var _GG = function(a, b, c, d, x, s, ac) {
		    a = addUnsigned(a, addUnsigned(addUnsigned(_G(b, c, d), x), ac));
		    return addUnsigned(rotateLeft(a, s), b);
		  };

		  var _HH = function(a, b, c, d, x, s, ac) {
		    a = addUnsigned(a, addUnsigned(addUnsigned(_H(b, c, d), x), ac));
		    return addUnsigned(rotateLeft(a, s), b);
		  };

		  var _II = function(a, b, c, d, x, s, ac) {
		    a = addUnsigned(a, addUnsigned(addUnsigned(_I(b, c, d), x), ac));
		    return addUnsigned(rotateLeft(a, s), b);
		  };

		  var convertToWordArray = function(str) {
		    var lWordCount;
		    var lMessageLength = str.length;
		    var lNumberOfWords_temp1 = lMessageLength + 8;
		    var lNumberOfWords_temp2 = (lNumberOfWords_temp1 - (lNumberOfWords_temp1 % 64)) / 64;
		    var lNumberOfWords = (lNumberOfWords_temp2 + 1) * 16;
		    var lWordArray = new Array(lNumberOfWords - 1);
		    var lBytePosition = 0;
		    var lByteCount = 0;
		    while (lByteCount < lMessageLength) {
		      lWordCount = (lByteCount - (lByteCount % 4)) / 4;
		      lBytePosition = (lByteCount % 4) * 8;
		      lWordArray[lWordCount] = (lWordArray[lWordCount] | (str.charCodeAt(lByteCount) << lBytePosition));
		      lByteCount++;
		    }
		    lWordCount = (lByteCount - (lByteCount % 4)) / 4;
		    lBytePosition = (lByteCount % 4) * 8;
		    lWordArray[lWordCount] = lWordArray[lWordCount] | (0x80 << lBytePosition);
		    lWordArray[lNumberOfWords - 2] = lMessageLength << 3;
		    lWordArray[lNumberOfWords - 1] = lMessageLength >>> 29;
		    return lWordArray;
		  };

		  var wordToHex = function(lValue) {
		    var wordToHexValue = '',
		      wordToHexValue_temp = '',
		      lByte, lCount;
		    for (lCount = 0; lCount <= 3; lCount++) {
		      lByte = (lValue >>> (lCount * 8)) & 255;
		      wordToHexValue_temp = '0' + lByte.toString(16);
		      wordToHexValue = wordToHexValue + wordToHexValue_temp.substr(wordToHexValue_temp.length - 2, 2);
		    }
		    return wordToHexValue;
		  };

		  var x = [],
		    k, AA, BB, CC, DD, a, b, c, d, S11 = 7,
		    S12 = 12,
		    S13 = 17,
		    S14 = 22,
		    S21 = 5,
		    S22 = 9,
		    S23 = 14,
		    S24 = 20,
		    S31 = 4,
		    S32 = 11,
		    S33 = 16,
		    S34 = 23,
		    S41 = 6,
		    S42 = 10,
		    S43 = 15,
		    S44 = 21;

		  str = XPRSHelper.utf8_encode(str);
		  x = convertToWordArray(str);
		  a = 0x67452301;
		  b = 0xEFCDAB89;
		  c = 0x98BADCFE;
		  d = 0x10325476;

		  xl = x.length;
		  for (k = 0; k < xl; k += 16) {
		    AA = a;
		    BB = b;
		    CC = c;
		    DD = d;
		    a = _FF(a, b, c, d, x[k + 0], S11, 0xD76AA478);
		    d = _FF(d, a, b, c, x[k + 1], S12, 0xE8C7B756);
		    c = _FF(c, d, a, b, x[k + 2], S13, 0x242070DB);
		    b = _FF(b, c, d, a, x[k + 3], S14, 0xC1BDCEEE);
		    a = _FF(a, b, c, d, x[k + 4], S11, 0xF57C0FAF);
		    d = _FF(d, a, b, c, x[k + 5], S12, 0x4787C62A);
		    c = _FF(c, d, a, b, x[k + 6], S13, 0xA8304613);
		    b = _FF(b, c, d, a, x[k + 7], S14, 0xFD469501);
		    a = _FF(a, b, c, d, x[k + 8], S11, 0x698098D8);
		    d = _FF(d, a, b, c, x[k + 9], S12, 0x8B44F7AF);
		    c = _FF(c, d, a, b, x[k + 10], S13, 0xFFFF5BB1);
		    b = _FF(b, c, d, a, x[k + 11], S14, 0x895CD7BE);
		    a = _FF(a, b, c, d, x[k + 12], S11, 0x6B901122);
		    d = _FF(d, a, b, c, x[k + 13], S12, 0xFD987193);
		    c = _FF(c, d, a, b, x[k + 14], S13, 0xA679438E);
		    b = _FF(b, c, d, a, x[k + 15], S14, 0x49B40821);
		    a = _GG(a, b, c, d, x[k + 1], S21, 0xF61E2562);
		    d = _GG(d, a, b, c, x[k + 6], S22, 0xC040B340);
		    c = _GG(c, d, a, b, x[k + 11], S23, 0x265E5A51);
		    b = _GG(b, c, d, a, x[k + 0], S24, 0xE9B6C7AA);
		    a = _GG(a, b, c, d, x[k + 5], S21, 0xD62F105D);
		    d = _GG(d, a, b, c, x[k + 10], S22, 0x2441453);
		    c = _GG(c, d, a, b, x[k + 15], S23, 0xD8A1E681);
		    b = _GG(b, c, d, a, x[k + 4], S24, 0xE7D3FBC8);
		    a = _GG(a, b, c, d, x[k + 9], S21, 0x21E1CDE6);
		    d = _GG(d, a, b, c, x[k + 14], S22, 0xC33707D6);
		    c = _GG(c, d, a, b, x[k + 3], S23, 0xF4D50D87);
		    b = _GG(b, c, d, a, x[k + 8], S24, 0x455A14ED);
		    a = _GG(a, b, c, d, x[k + 13], S21, 0xA9E3E905);
		    d = _GG(d, a, b, c, x[k + 2], S22, 0xFCEFA3F8);
		    c = _GG(c, d, a, b, x[k + 7], S23, 0x676F02D9);
		    b = _GG(b, c, d, a, x[k + 12], S24, 0x8D2A4C8A);
		    a = _HH(a, b, c, d, x[k + 5], S31, 0xFFFA3942);
		    d = _HH(d, a, b, c, x[k + 8], S32, 0x8771F681);
		    c = _HH(c, d, a, b, x[k + 11], S33, 0x6D9D6122);
		    b = _HH(b, c, d, a, x[k + 14], S34, 0xFDE5380C);
		    a = _HH(a, b, c, d, x[k + 1], S31, 0xA4BEEA44);
		    d = _HH(d, a, b, c, x[k + 4], S32, 0x4BDECFA9);
		    c = _HH(c, d, a, b, x[k + 7], S33, 0xF6BB4B60);
		    b = _HH(b, c, d, a, x[k + 10], S34, 0xBEBFBC70);
		    a = _HH(a, b, c, d, x[k + 13], S31, 0x289B7EC6);
		    d = _HH(d, a, b, c, x[k + 0], S32, 0xEAA127FA);
		    c = _HH(c, d, a, b, x[k + 3], S33, 0xD4EF3085);
		    b = _HH(b, c, d, a, x[k + 6], S34, 0x4881D05);
		    a = _HH(a, b, c, d, x[k + 9], S31, 0xD9D4D039);
		    d = _HH(d, a, b, c, x[k + 12], S32, 0xE6DB99E5);
		    c = _HH(c, d, a, b, x[k + 15], S33, 0x1FA27CF8);
		    b = _HH(b, c, d, a, x[k + 2], S34, 0xC4AC5665);
		    a = _II(a, b, c, d, x[k + 0], S41, 0xF4292244);
		    d = _II(d, a, b, c, x[k + 7], S42, 0x432AFF97);
		    c = _II(c, d, a, b, x[k + 14], S43, 0xAB9423A7);
		    b = _II(b, c, d, a, x[k + 5], S44, 0xFC93A039);
		    a = _II(a, b, c, d, x[k + 12], S41, 0x655B59C3);
		    d = _II(d, a, b, c, x[k + 3], S42, 0x8F0CCC92);
		    c = _II(c, d, a, b, x[k + 10], S43, 0xFFEFF47D);
		    b = _II(b, c, d, a, x[k + 1], S44, 0x85845DD1);
		    a = _II(a, b, c, d, x[k + 8], S41, 0x6FA87E4F);
		    d = _II(d, a, b, c, x[k + 15], S42, 0xFE2CE6E0);
		    c = _II(c, d, a, b, x[k + 6], S43, 0xA3014314);
		    b = _II(b, c, d, a, x[k + 13], S44, 0x4E0811A1);
		    a = _II(a, b, c, d, x[k + 4], S41, 0xF7537E82);
		    d = _II(d, a, b, c, x[k + 11], S42, 0xBD3AF235);
		    c = _II(c, d, a, b, x[k + 2], S43, 0x2AD7D2BB);
		    b = _II(b, c, d, a, x[k + 9], S44, 0xEB86D391);
		    a = addUnsigned(a, AA);
		    b = addUnsigned(b, BB);
		    c = addUnsigned(c, CC);
		    d = addUnsigned(d, DD);
		  }

		  var temp = wordToHex(a) + wordToHex(b) + wordToHex(c) + wordToHex(d);

		  return temp.toLowerCase();
		
};



XPRSHelper.isTransparentColor = function(color){
	if (color === undefined){
		return false;
	}
	var isTransparent = color == "rgba(0, 0, 0, 0)" || color == "transparent";
	if (color.indexOf("rgba") != -1 && color.indexOf(", 0)") != -1){
		isTransparent = true;
	}
	return isTransparent
};

XPRSHelper.checkBrowserSupport = function(){
	var currentBrowser = XPRSHelper.getBrowser();
	if (currentBrowser.toLowerCase().indexOf("chrome") == -1){
		//We does not support your browser for the time being please download chrome
	}
};

XPRSHelper.getBrowser = function(){
	var ua= navigator.userAgent, tem, 
	M= ua.match(/(opera|chrome|safari|firefox|msie|trident(?=\/))\/?\s*(\d+)/i) || [];
	if(/trident/i.test(M[1])){
	    tem=  /\brv[ :]+(\d+)/g.exec(ua) || [];
	    return 'IE '+(tem[1] || '');
	}
	if(M[1]=== 'Chrome'){
	    tem= ua.match(/\bOPR\/(\d+)/);
	    if(tem!= null) return 'Opera '+tem[1];
	}
	M= M[2]? [M[1], M[2]]: [navigator.appName, navigator.appVersion, '-?'];
	if((tem= ua.match(/version\/(\d+)/i))!= null) M.splice(1, 1, tem[1]);
	return M.join(' ');
};

//
//XPRSHelper.MD5 = function(s){
//	    function L(k, d) {
//	        return (k << d) | (k >>> (32 - d))
//	    }
//
//	    function K(G, k) {
//	        var I, d, F, H, x;
//	        F = (G & 2147483648);
//	        H = (k & 2147483648);
//	        I = (G & 1073741824);
//	        d = (k & 1073741824);
//	        x = (G & 1073741823) + (k & 1073741823);
//	        if (I & d) {
//	            return (x ^ 2147483648 ^ F ^ H)
//	        }
//	        if (I | d) {
//	            if (x & 1073741824) {
//	                return (x ^ 3221225472 ^ F ^ H)
//	            } else {
//	                return (x ^ 1073741824 ^ F ^ H)
//	            }
//	        } else {
//	            return (x ^ F ^ H)
//	        }
//	    }
//
//	    function r(d, F, k) {
//	        return (d & F) | ((~d) & k)
//	    }
//
//	    function q(d, F, k) {
//	        return (d & k) | (F & (~k))
//	    }
//
//	    function p(d, F, k) {
//	        return (d ^ F ^ k)
//	    }
//
//	    function n(d, F, k) {
//	        return (F ^ (d | (~k)))
//	    }
//
//	    function u(G, F, aa, Z, k, H, I) {
//	        G = K(G, K(K(r(F, aa, Z), k), I));
//	        return K(L(G, H), F)
//	    }
//
//	    function f(G, F, aa, Z, k, H, I) {
//	        G = K(G, K(K(q(F, aa, Z), k), I));
//	        return K(L(G, H), F)
//	    }
//
//	    function D(G, F, aa, Z, k, H, I) {
//	        G = K(G, K(K(p(F, aa, Z), k), I));
//	        return K(L(G, H), F)
//	    }
//
//	    function t(G, F, aa, Z, k, H, I) {
//	        G = K(G, K(K(n(F, aa, Z), k), I));
//	        return K(L(G, H), F)
//	    }
//
//	    function e(G) {
//	        var Z;
//	        var F = G.length;
//	        var x = F + 8;
//	        var k = (x - (x % 64)) / 64;
//	        var I = (k + 1) * 16;
//	        var aa = Array(I - 1);
//	        var d = 0;
//	        var H = 0;
//	        while (H < F) {
//	            Z = (H - (H % 4)) / 4;
//	            d = (H % 4) * 8;
//	            aa[Z] = (aa[Z] | (G.charCodeAt(H) << d));
//	            H++
//	        }
//	        Z = (H - (H % 4)) / 4;
//	        d = (H % 4) * 8;
//	        aa[Z] = aa[Z] | (128 << d);
//	        aa[I - 2] = F << 3;
//	        aa[I - 1] = F >>> 29;
//	        return aa
//	    }
//
//	    function B(x) {
//	        var k = "",
//	            F = "",
//	            G, d;
//	        for (d = 0; d <= 3; d++) {
//	            G = (x >>> (d * 8)) & 255;
//	            F = "0" + G.toString(16);
//	            k = k + F.substr(F.length - 2, 2)
//	        }
//	        return k
//	    }
//
//	    function J(k) {
//	        k = k.replace(/rn/g, "n");
//	        var d = "";
//	        for (var F = 0; F < k.length; F++) {
//	            var x = k.charCodeAt(F);
//	            if (x < 128) {
//	                d += String.fromCharCode(x)
//	            } else {
//	                if ((x > 127) && (x < 2048)) {
//	                    d += String.fromCharCode((x >> 6) | 192);
//	                    d += String.fromCharCode((x & 63) | 128)
//	                } else {
//	                    d += String.fromCharCode((x >> 12) | 224);
//	                    d += String.fromCharCode(((x >> 6) & 63) | 128);
//	                    d += String.fromCharCode((x & 63) | 128)
//	                }
//	            }
//	        }
//	        return d
//	    }
//	    var C = Array();
//	    var P, h, E, v, g, Y, X, W, V;
//	    var S = 7,
//	        Q = 12,
//	        N = 17,
//	        M = 22;
//	    var A = 5,
//	        z = 9,
//	        y = 14,
//	        w = 20;
//	    var o = 4,
//	        m = 11,
//	        l = 16,
//	        j = 23;
//	    var U = 6,
//	        T = 10,
//	        R = 15,
//	        O = 21;
//	    s = J(s);
//	    C = e(s);
//	    Y = 1732584193;
//	    X = 4023233417;
//	    W = 2562383102;
//	    V = 271733878;
//	    for (P = 0; P < C.length; P += 16) {
//	        h = Y;
//	        E = X;
//	        v = W;
//	        g = V;
//	        Y = u(Y, X, W, V, C[P + 0], S, 3614090360);
//	        V = u(V, Y, X, W, C[P + 1], Q, 3905402710);
//	        W = u(W, V, Y, X, C[P + 2], N, 606105819);
//	        X = u(X, W, V, Y, C[P + 3], M, 3250441966);
//	        Y = u(Y, X, W, V, C[P + 4], S, 4118548399);
//	        V = u(V, Y, X, W, C[P + 5], Q, 1200080426);
//	        W = u(W, V, Y, X, C[P + 6], N, 2821735955);
//	        X = u(X, W, V, Y, C[P + 7], M, 4249261313);
//	        Y = u(Y, X, W, V, C[P + 8], S, 1770035416);
//	        V = u(V, Y, X, W, C[P + 9], Q, 2336552879);
//	        W = u(W, V, Y, X, C[P + 10], N, 4294925233);
//	        X = u(X, W, V, Y, C[P + 11], M, 2304563134);
//	        Y = u(Y, X, W, V, C[P + 12], S, 1804603682);
//	        V = u(V, Y, X, W, C[P + 13], Q, 4254626195);
//	        W = u(W, V, Y, X, C[P + 14], N, 2792965006);
//	        X = u(X, W, V, Y, C[P + 15], M, 1236535329);
//	        Y = f(Y, X, W, V, C[P + 1], A, 4129170786);
//	        V = f(V, Y, X, W, C[P + 6], z, 3225465664);
//	        W = f(W, V, Y, X, C[P + 11], y, 643717713);
//	        X = f(X, W, V, Y, C[P + 0], w, 3921069994);
//	        Y = f(Y, X, W, V, C[P + 5], A, 3593408605);
//	        V = f(V, Y, X, W, C[P + 10], z, 38016083);
//	        W = f(W, V, Y, X, C[P + 15], y, 3634488961);
//	        X = f(X, W, V, Y, C[P + 4], w, 3889429448);
//	        Y = f(Y, X, W, V, C[P + 9], A, 568446438);
//	        V = f(V, Y, X, W, C[P + 14], z, 3275163606);
//	        W = f(W, V, Y, X, C[P + 3], y, 4107603335);
//	        X = f(X, W, V, Y, C[P + 8], w, 1163531501);
//	        Y = f(Y, X, W, V, C[P + 13], A, 2850285829);
//	        V = f(V, Y, X, W, C[P + 2], z, 4243563512);
//	        W = f(W, V, Y, X, C[P + 7], y, 1735328473);
//	        X = f(X, W, V, Y, C[P + 12], w, 2368359562);
//	        Y = D(Y, X, W, V, C[P + 5], o, 4294588738);
//	        V = D(V, Y, X, W, C[P + 8], m, 2272392833);
//	        W = D(W, V, Y, X, C[P + 11], l, 1839030562);
//	        X = D(X, W, V, Y, C[P + 14], j, 4259657740);
//	        Y = D(Y, X, W, V, C[P + 1], o, 2763975236);
//	        V = D(V, Y, X, W, C[P + 4], m, 1272893353);
//	        W = D(W, V, Y, X, C[P + 7], l, 4139469664);
//	        X = D(X, W, V, Y, C[P + 10], j, 3200236656);
//	        Y = D(Y, X, W, V, C[P + 13], o, 681279174);
//	        V = D(V, Y, X, W, C[P + 0], m, 3936430074);
//	        W = D(W, V, Y, X, C[P + 3], l, 3572445317);
//	        X = D(X, W, V, Y, C[P + 6], j, 76029189);
//	        Y = D(Y, X, W, V, C[P + 9], o, 3654602809);
//	        V = D(V, Y, X, W, C[P + 12], m, 3873151461);
//	        W = D(W, V, Y, X, C[P + 15], l, 530742520);
//	        X = D(X, W, V, Y, C[P + 2], j, 3299628645);
//	        Y = t(Y, X, W, V, C[P + 0], U, 4096336452);
//	        V = t(V, Y, X, W, C[P + 7], T, 1126891415);
//	        W = t(W, V, Y, X, C[P + 14], R, 2878612391);
//	        X = t(X, W, V, Y, C[P + 5], O, 4237533241);
//	        Y = t(Y, X, W, V, C[P + 12], U, 1700485571);
//	        V = t(V, Y, X, W, C[P + 3], T, 2399980690);
//	        W = t(W, V, Y, X, C[P + 10], R, 4293915773);
//	        X = t(X, W, V, Y, C[P + 1], O, 2240044497);
//	        Y = t(Y, X, W, V, C[P + 8], U, 1873313359);
//	        V = t(V, Y, X, W, C[P + 15], T, 4264355552);
//	        W = t(W, V, Y, X, C[P + 6], R, 2734768916);
//	        X = t(X, W, V, Y, C[P + 13], O, 1309151649);
//	        Y = t(Y, X, W, V, C[P + 4], U, 4149444226);
//	        V = t(V, Y, X, W, C[P + 11], T, 3174756917);
//	        W = t(W, V, Y, X, C[P + 2], R, 718787259);
//	        X = t(X, W, V, Y, C[P + 9], O, 3951481745);
//	        Y = K(Y, h);
//	        X = K(X, E);
//	        W = K(W, v);
//	        V = K(V, g)
//	    }
//	    var i = B(Y) + B(X) + B(W) + B(V);
//	    return i.toLowerCase()
//};



XPRSHelper.slugify = function(text){
  return text.toString().toLowerCase()
    .replace(/\s+/g, '-')           // Replace spaces with -
    .replace(/[^\w\-]+/g, '')       // Remove all non-word chars
    .replace(/\-\-+/g, '-')         // Replace multiple - with single -
    .replace(/^-+/, '')             // Trim - from start of text
    .replace(/-+$/, '');            // Trim - from end of text
}

XPRSHelper.isChrome = function(){
	var isOpera = !!window.opera || navigator.userAgent.indexOf(' OPR/') >= 0;
    // Opera 8.0+ (UA detection to detect Blink/v8-powered Opera)
	//var isFirefox = typeof InstallTrigger !== 'undefined';   // Firefox 1.0+
	//var isSafari = Object.prototype.toString.call(window.HTMLElement).indexOf('Constructor') > 0;
	    // At least Safari 3+: "[object HTMLElementConstructor]"
	var isChrome = !!window.chrome && !isOpera;     // Chrome 1+ 
	return isChrome;
	//var isIE = /*@cc_on!@*/false || !!document.documentMode; // At least IE6
}

XPRSHelper.utf8_encode = function (argString) {
	  //  discuss at: http://phpjs.org/functions/utf8_encode/
	  // original by: Webtoolkit.info (http://www.webtoolkit.info/)
	  // improved by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
	  // improved by: sowberry
	  // improved by: Jack
	  // improved by: Yves Sucaet
	  // improved by: kirilloid
	  // bugfixed by: Onno Marsman
	  // bugfixed by: Onno Marsman
	  // bugfixed by: Ulrich
	  // bugfixed by: Rafal Kukawski
	  // bugfixed by: kirilloid
	  //   example 1: utf8_encode('Kevin van Zonneveld');
	  //   returns 1: 'Kevin van Zonneveld'

	  if (argString === null || typeof argString === 'undefined') {
	    return '';
	  }

	  var string = (argString + ''); // .replace(/\r\n/g, "\n").replace(/\r/g, "\n");
	  var utftext = '',
	    start, end, stringl = 0;

	  start = end = 0;
	  stringl = string.length;
	  for (var n = 0; n < stringl; n++) {
	    var c1 = string.charCodeAt(n);
	    var enc = null;

	    if (c1 < 128) {
	      end++;
	    } else if (c1 > 127 && c1 < 2048) {
	      enc = String.fromCharCode(
	        (c1 >> 6) | 192, (c1 & 63) | 128
	      );
	    } else if (c1 & 0xF800 != 0xD800) {
	      enc = String.fromCharCode(
	        (c1 >> 12) | 224, ((c1 >> 6) & 63) | 128, (c1 & 63) | 128
	      );
	    } else { // surrogate pairs
	      if (c1 & 0xFC00 != 0xD800) {
	        throw new RangeError('Unmatched trail surrogate at ' + n);
	      }
	      var c2 = string.charCodeAt(++n);
	      if (c2 & 0xFC00 != 0xDC00) {
	        throw new RangeError('Unmatched lead surrogate at ' + (n - 1));
	      }
	      c1 = ((c1 & 0x3FF) << 10) + (c2 & 0x3FF) + 0x10000;
	      enc = String.fromCharCode(
	        (c1 >> 18) | 240, ((c1 >> 12) & 63) | 128, ((c1 >> 6) & 63) | 128, (c1 & 63) | 128
	      );
	    }
	    if (enc !== null) {
	      if (end > start) {
	        utftext += string.slice(start, end);
	      }
	      utftext += enc;
	      start = end = n + 1;
	    }
	  }

	  if (end > start) {
	    utftext += string.slice(start, stringl);
	  }

	  return utftext;
	};
	
	/*
	 * Date Format 1.2.3
	 * (c) 2007-2009 Steven Levithan <stevenlevithan.com>
	 * MIT license
	 *
	 * Includes enhancements by Scott Trenda <scott.trenda.net>
	 * and Kris Kowal <cixar.com/~kris.kowal/>
	 *
	 * Accepts a date, a mask, or a date and a mask.
	 * Returns a formatted version of the given date.
	 * The date defaults to the current date/time.
	 * The mask defaults to dateFormat.masks.default.
	 */

	var dateFormat = function () {
		var	token = /d{1,4}|m{1,4}|yy(?:yy)?|([HhMsTt])\1?|[LloSZ]|"[^"]*"|'[^']*'/g,
			timezone = /\b(?:[PMCEA][SDP]T|(?:Pacific|Mountain|Central|Eastern|Atlantic) (?:Standard|Daylight|Prevailing) Time|(?:GMT|UTC)(?:[-+]\d{4})?)\b/g,
			timezoneClip = /[^-+\dA-Z]/g,
			pad = function (val, len) {
				val = String(val);
				len = len || 2;
				while (val.length < len) val = "0" + val;
				return val;
			};

		// Regexes and supporting functions are cached through closure
		return function (date, mask, utc) {
			var dF = dateFormat;

			// You can't provide utc if you skip other args (use the "UTC:" mask prefix)
			if (arguments.length == 1 && Object.prototype.toString.call(date) == "[object String]" && !/\d/.test(date)) {
				mask = date;
				date = undefined;
			}

			// Passing date through Date applies Date.parse, if necessary
			date = date ? new Date(date) : new Date;
			if (isNaN(date)) throw SyntaxError("invalid date");

			mask = String(dF.masks[mask] || mask || dF.masks["default"]);

			// Allow setting the utc argument via the mask
			if (mask.slice(0, 4) == "UTC:") {
				mask = mask.slice(4);
				utc = true;
			}

			var	_ = utc ? "getUTC" : "get",
				d = date[_ + "Date"](),
				D = date[_ + "Day"](),
				m = date[_ + "Month"](),
				y = date[_ + "FullYear"](),
				H = date[_ + "Hours"](),
				M = date[_ + "Minutes"](),
				s = date[_ + "Seconds"](),
				L = date[_ + "Milliseconds"](),
				o = utc ? 0 : date.getTimezoneOffset(),
				flags = {
					d:    d,
					dd:   pad(d),
					ddd:  dF.i18n.dayNames[D],
					dddd: dF.i18n.dayNames[D + 7],
					m:    m + 1,
					mm:   pad(m + 1),
					mmm:  dF.i18n.monthNames[m],
					mmmm: dF.i18n.monthNames[m + 12],
					yy:   String(y).slice(2),
					yyyy: y,
					h:    H % 12 || 12,
					hh:   pad(H % 12 || 12),
					H:    H,
					HH:   pad(H),
					M:    M,
					MM:   pad(M),
					s:    s,
					ss:   pad(s),
					l:    pad(L, 3),
					L:    pad(L > 99 ? Math.round(L / 10) : L),
					t:    H < 12 ? "a"  : "p",
					tt:   H < 12 ? "am" : "pm",
					T:    H < 12 ? "A"  : "P",
					TT:   H < 12 ? "AM" : "PM",
					Z:    utc ? "UTC" : (String(date).match(timezone) || [""]).pop().replace(timezoneClip, ""),
					o:    (o > 0 ? "-" : "+") + pad(Math.floor(Math.abs(o) / 60) * 100 + Math.abs(o) % 60, 4),
					S:    ["th", "st", "nd", "rd"][d % 10 > 3 ? 0 : (d % 100 - d % 10 != 10) * d % 10]
				};

			return mask.replace(token, function ($0) {
				return $0 in flags ? flags[$0] : $0.slice(1, $0.length - 1);
			});
		};
	}();

	// Some common format strings
	dateFormat.masks = {
		"default":      "ddd mmm dd yyyy HH:MM:ss",
		shortDate:      "m/d/yy",
		mediumDate:     "mmm d, yyyy",
		longDate:       "mmmm d, yyyy",
		fullDate:       "dddd, mmmm d, yyyy",
		shortTime:      "h:MM TT",
		mediumTime:     "h:MM:ss TT",
		longTime:       "h:MM:ss TT Z",
		isoDate:        "yyyy-mm-dd",
		isoTime:        "HH:MM:ss",
		isoDateTime:    "yyyy-mm-dd'T'HH:MM:ss",
		isoUtcDateTime: "UTC:yyyy-mm-dd'T'HH:MM:ss'Z'"
	};

	// Internationalization strings
	dateFormat.i18n = {
		dayNames: [
			"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
			"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
		],
		monthNames: [
			"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
			"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"
		]
	};

	// For convenience...
	Date.prototype.format = function (mask, utc) {
		return dateFormat(this, mask, utc);
	};


	XPRSHelper.shortUuid = function(prefix) {
		if (typeof prefix == "undefined"){
			prefix = "element";
		}
		return prefix + '-xxxxxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
			var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
			return v.toString(16);
		});
	};






