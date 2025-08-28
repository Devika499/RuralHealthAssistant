import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    // Load chat history when component mounts
    loadChatHistory();
  }, []);

  const loadChatHistory = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API_URL}/chat/history`);
      setMessages(response.data);
    } catch (err) {
      setError('Failed to load chat history');
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async (text) => {
    try {
      setIsLoading(true);
      const response = await axios.post(`${API_URL}/chat/`, {
        message: text,
      });

      // Add user message to messages array
      const userMessage = {
        text,
        sender: 'user',
        timestamp: new Date().toISOString(),
        language: 'hi', // Default to Hindi
      };

      // Add bot response to messages array
      const botMessage = {
        text: response.data.response,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        language: 'hi',
      };

      setMessages(prev => [...prev, userMessage, botMessage]);
    } catch (err) {
      setError('Failed to send message');
    } finally {
      setIsLoading(false);
    }
  };

  const clearMessages = () => setMessages([]);

  return { messages, isLoading, error, sendMessage, clearMessages };
};
