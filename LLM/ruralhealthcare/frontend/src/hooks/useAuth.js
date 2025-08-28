import { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const API_URL = 'http://localhost:8000';

export const useAuth = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      // Fetch user data here if needed
      setUser({
        id: localStorage.getItem('userId'),
        name: localStorage.getItem('userName'),
        mobileNumber: localStorage.getItem('userMobile'),
      });
    }
    setLoading(false);
  }, []);

  const login = async (mobileNumber, password) => {
    try {
      const response = await axios.post(`${API_URL}/token`, {
        username: mobileNumber,
        password,
      });

      const { access_token } = response.data;
      localStorage.setItem('token', access_token);
      localStorage.setItem('userMobile', mobileNumber);
      
      // Set Authorization header for subsequent requests
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      // Fetch user details
      const userResponse = await axios.get(`${API_URL}/users/me`);
      const userData = userResponse.data;
      localStorage.setItem('userId', userData.id);
      localStorage.setItem('userName', userData.name);
      
      setUser(userData);
      return true;
    } catch (error) {
      console.error('Login error:', error);
      throw new Error('Login failed');
    }
  };

  const register = async (userData) => {
    try {
      await axios.post(`${API_URL}/users/`, userData);
      navigate('/login');
    } catch (error) {
      throw new Error('Registration failed');
    }
  };

  const updateUser = async (updates) => {
    try {
      await axios.patch(`${API_URL}/users/${user.id}`, updates);
      setUser({ ...user, ...updates });
      // Update local storage if needed
    } catch (error) {
      throw new Error('Failed to update profile');
    }
  };

  // Accepts chatMessages and clearMessages as arguments
  const logout = async (chatMessages = [], clearMessages) => {
    try {
      if (chatMessages && chatMessages.length > 0) {
        await axios.post(`${API_URL}/save-chat-to-history`, chatMessages);
      }
    } catch (e) {
      // Ignore errors for now, but could show a notification
      console.error('Failed to save chat to medical history:', e);
    }
    localStorage.removeItem('token');
    localStorage.removeItem('userId');
    localStorage.removeItem('userName');
    localStorage.removeItem('userMobile');
    delete axios.defaults.headers.common['Authorization'];
    setUser(null);
    if (typeof clearMessages === 'function') clearMessages();
    navigate('/login');
  };

  return { user, login, register, updateUser, logout, loading };
};
